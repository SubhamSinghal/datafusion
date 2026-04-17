// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`PushDownTopKThroughJoin`] pushes TopK (Sort with fetch) through outer joins
//!
//! When a `Sort` with a fetch limit sits above an outer join and all sort
//! expressions come from the **preserved** side, this rule inserts a copy
//! of the `Sort(fetch)` on that input to reduce the number of rows
//! entering the join.
//!
//! This is correct because:
//! - A LEFT JOIN preserves every left row (each appears at least once in the
//!   output). The final top-N by left-side columns must come from the top-N
//!   left rows.
//! - The same reasoning applies symmetrically for RIGHT JOIN and right-side
//!   columns.
//!
//! The top-level sort is kept for correctness since a 1-to-many join can
//! produce more than N output rows from N input rows.
//!
//! ## Example
//!
//! Before:
//! ```text
//! Sort: t1.b ASC, fetch=3
//!   Left Join: t1.a = t2.c
//!     Scan: t1     ← scans ALL rows
//!     Scan: t2
//! ```
//!
//! After:
//! ```text
//! Sort: t1.b ASC, fetch=3
//!   Left Join: t1.a = t2.c
//!     Sort: t1.b ASC, fetch=3  ← pushed down
//!       Scan: t1
//!     Scan: t2
//! ```

use std::sync::Arc;

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

use crate::utils::{has_all_column_refs, schema_columns};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Column, Result};
use datafusion_expr::logical_plan::{
    JoinType, LogicalPlan, Projection, Sort as SortPlan,
};
use datafusion_expr::{Expr, SortExpr};

/// Optimization rule that pushes TopK (Sort with fetch) through
/// LEFT / RIGHT outer joins when all sort expressions come from
/// the preserved side.
///
/// See module-level documentation for details.
#[derive(Default, Debug)]
pub struct PushDownTopKThroughJoin;

impl PushDownTopKThroughJoin {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for PushDownTopKThroughJoin {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match Sort with fetch (TopK)
        let LogicalPlan::Sort(sort) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(fetch) = sort.fetch else {
            return Ok(Transformed::no(plan));
        };

        // Don't push if any sort expression is non-deterministic (e.g. random()).
        // Duplicating such expressions would produce different values at each
        // evaluation point, potentially changing the result.
        if sort.expr.iter().any(|se| se.expr.is_volatile()) {
            return Ok(Transformed::no(plan));
        }

        // Check if the child is a Join (look through Projection)
        let (has_projection, join) = match sort.input.as_ref() {
            LogicalPlan::Join(join) => (false, join),
            LogicalPlan::Projection(proj) => match proj.input.as_ref() {
                LogicalPlan::Join(join) => (true, join),
                _ => return Ok(Transformed::no(plan)),
            },
            _ => return Ok(Transformed::no(plan)),
        };

        // Only outer joins where the preserved side is known.
        // Semi/Anti joins are excluded: not all preserved-side rows appear in
        // the output (only matched/unmatched rows do), so pushing fetch=N to
        // the preserved child can drop rows that would have survived the filter.
        // No non-equijoin filter (conservative — filter may change row count).
        let preserved_is_left = match join.join_type {
            JoinType::Left => true,
            JoinType::Right => false,
            _ => return Ok(Transformed::no(plan)),
        };
        if join.filter.is_some() {
            return Ok(Transformed::no(plan));
        }

        // Check all sort expression columns come from the preserved side.
        // When there's a projection, resolve sort expressions through it first
        // since the sort references post-projection columns.
        let resolved_sort_exprs = if has_projection {
            let LogicalPlan::Projection(proj) = sort.input.as_ref() else {
                unreachable!()
            };
            resolve_sort_exprs_through_projection(&sort.expr, proj)?
        } else {
            sort.expr.clone()
        };

        let preserved_schema = if preserved_is_left {
            join.left.schema()
        } else {
            join.right.schema()
        };
        let preserved_cols = schema_columns(preserved_schema);

        let all_from_preserved = resolved_sort_exprs
            .iter()
            .all(|sort_expr| has_all_column_refs(&sort_expr.expr, &preserved_cols));
        if !all_from_preserved {
            return Ok(Transformed::no(plan));
        }

        // Push through when the preserved child has no Sort, or has a Sort
        // with a larger/no fetch limit (our tighter limit reduces data further).
        //
        // Example (push): Sort(a ASC, fetch=5) → Join → Sort(a ASC, fetch=10)
        //   Child limits to 10, our tighter fetch=5 reduces data further.
        //
        // Example (push): Sort(a ASC, fetch=5) → Join → Sort(a ASC)
        //   Child has no fetch (full sort), adding fetch=5 limits early.
        //
        // Example (skip): Sort(a ASC, fetch=5) → Join → Sort(a ASC, fetch=3)
        //   Child already limits to 3 rows, pushing fetch=5 won't help.
        let preserved_child = if preserved_is_left {
            &join.left
        } else {
            &join.right
        };
        if let LogicalPlan::Sort(child_sort) = preserved_child.as_ref() {
            // Compare using resolved expressions since the parent sort may
            // reference post-projection column names while the child uses
            // pre-projection expressions.
            let same_exprs = child_sort.expr == resolved_sort_exprs;
            let child_fetch_tighter = match child_sort.fetch {
                Some(child_fetch) => child_fetch <= fetch,
                None => false,
            };
            if same_exprs && child_fetch_tighter {
                return Ok(Transformed::no(plan));
            }
        }

        // Create the new Sort(fetch) on the preserved child.
        // Use the resolved expressions (pre-projection) for the pushed Sort.
        //
        // If the child is already a Sort with the same expressions but a larger
        // fetch, tighten its fetch in-place instead of stacking a redundant Sort
        // on top.
        let (sort_input, sort_exprs) = match preserved_child.as_ref() {
            LogicalPlan::Sort(child_sort) if child_sort.expr == resolved_sort_exprs => {
                (Arc::clone(&child_sort.input), child_sort.expr.clone())
            }
            _ => (Arc::clone(preserved_child), resolved_sort_exprs),
        };
        let new_child_sort = Arc::new(LogicalPlan::Sort(SortPlan {
            expr: sort_exprs,
            input: sort_input,
            fetch: Some(fetch),
        }));

        // Reconstruct the join with the new child
        let mut new_join = join.clone();
        if preserved_is_left {
            new_join.left = new_child_sort;
        } else {
            new_join.right = new_child_sort;
        }

        // Rebuild the tree: join → optional projection → top-level sort
        let new_join_plan = LogicalPlan::Join(new_join);
        let new_sort_input = if has_projection {
            // Reconstruct the Projection with the new join
            let LogicalPlan::Projection(proj) = sort.input.as_ref() else {
                unreachable!()
            };
            let mut new_proj = proj.clone();
            new_proj.input = Arc::new(new_join_plan);
            Arc::new(LogicalPlan::Projection(new_proj))
        } else {
            Arc::new(new_join_plan)
        };

        Ok(Transformed::yes(LogicalPlan::Sort(SortPlan {
            expr: sort.expr.clone(),
            input: new_sort_input,
            fetch: sort.fetch,
        })))
    }

    fn name(&self) -> &str {
        "push_down_topk_through_join"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
}

/// Resolve sort expressions through a projection by replacing column
/// references with the underlying projection expressions.
///
/// For example, if sort expr is `b ASC` and projection has `-t1.b AS b`,
/// the resolved sort expr becomes `-t1.b ASC`.
///
/// Before:
/// ```text
/// Sort: b ASC, fetch=3
///   Projection: -t1.b AS b
///     Join
///       t1
///       t2
/// ```
///
/// After resolving, the pushed Sort uses pre-projection expressions:
/// ```text
/// Sort: b ASC, fetch=3
///   Projection: -t1.b AS b
///     Join
///       Sort: -t1.b ASC, fetch=3  ← resolved through projection
///         t1
///       t2
/// ```
fn resolve_sort_exprs_through_projection(
    sort_exprs: &[SortExpr],
    projection: &Projection,
) -> Result<Vec<SortExpr>> {
    // Build map: output column name → underlying expression
    let replace_map: std::collections::HashMap<String, Expr> = projection
        .schema
        .iter()
        .zip(projection.expr.iter())
        .map(|((qualifier, field), expr)| {
            let key = Column::from((qualifier, field)).flat_name();
            (key, expr.clone().unalias())
        })
        .collect();

    sort_exprs
        .iter()
        .map(|sort_expr| {
            let new_expr = sort_expr.expr.clone().transform(|expr| {
                let replacement = match &expr {
                    Expr::Column(col) => replace_map.get(&col.flat_name()).cloned(),
                    _ => None,
                };
                Ok(replacement.map_or_else(|| Transformed::no(expr), Transformed::yes))
            })?;
            Ok(SortExpr {
                expr: new_expr.data,
                ..*sort_expr
            })
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::OptimizerContext;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::*;

    use datafusion_expr::col;
    use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> = vec![Arc::new(PushDownTopKThroughJoin::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    /// TopK on left-side columns above a LEFT JOIN → pushed to left child.
    #[test]
    fn topk_pushed_to_left_of_left_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          Left Join: t1.a = t2.a
            Sort: t1.b ASC NULLS LAST, fetch=3
              TableScan: t1
            TableScan: t2
        "
        )
    }

    /// TopK on right-side columns above a RIGHT JOIN → pushed to right child.
    #[test]
    fn topk_pushed_to_right_of_right_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Right,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t2.b").sort(true, false)], Some(5))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t2.b ASC NULLS LAST, fetch=5
          Right Join: t1.a = t2.a
            TableScan: t1
            Sort: t2.b ASC NULLS LAST, fetch=5
              TableScan: t2
        "
        )
    }

    /// TopK pushed through a Projection between Sort and Join.
    #[test]
    fn topk_pushed_through_projection() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .project(vec![col("t1.a"), col("t1.b"), col("t2.c")])?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          Projection: t1.a, t1.b, t2.c
            Left Join: t1.a = t2.a
              Sort: t1.b ASC NULLS LAST, fetch=3
                TableScan: t1
              TableScan: t2
        "
        )
    }

    /// INNER JOIN → no pushdown.
    #[test]
    fn topk_not_pushed_for_inner_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Inner,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          Inner Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// LEFT JOIN but sort on right-side columns → no pushdown.
    #[test]
    fn topk_not_pushed_for_wrong_side() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t2.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t2.b ASC NULLS LAST, fetch=3
          Left Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// Join with a non-equijoin filter → no pushdown (conservative).
    #[test]
    fn topk_not_pushed_with_join_filter() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                vec![col("t1.a").eq(col("t2.a"))],
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          Left Join:  Filter: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// Sort without fetch (unbounded) → no pushdown.
    #[test]
    fn topk_not_pushed_without_fetch() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort(vec![col("t1.b").sort(true, false)])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST
          Left Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// LEFT SEMI JOIN: pushing fetch is unsafe (not all left rows appear in output).
    #[test]
    fn topk_not_pushed_for_left_semi_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::LeftSemi,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          LeftSemi Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// LEFT ANTI JOIN: pushing fetch is unsafe (not all left rows appear in output).
    #[test]
    fn topk_not_pushed_for_left_anti_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::LeftAnti,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          LeftAnti Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// RIGHT SEMI JOIN: pushing fetch is unsafe (not all right rows appear in output).
    #[test]
    fn topk_not_pushed_for_right_semi_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::RightSemi,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t2.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t2.b ASC NULLS LAST, fetch=3
          RightSemi Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// RIGHT ANTI JOIN: pushing fetch is unsafe (not all right rows appear in output).
    #[test]
    fn topk_not_pushed_for_right_anti_join() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::RightAnti,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t2.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t2.b ASC NULLS LAST, fetch=3
          RightAnti Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// Multi-column sort with columns from both sides → no pushdown.
    #[test]
    fn topk_not_pushed_for_mixed_side_sort() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(
                vec![col("t1.b").sort(true, false), col("t2.b").sort(true, false)],
                Some(3),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, t2.b ASC NULLS LAST, fetch=3
          Left Join: t1.a = t2.a
            TableScan: t1
            TableScan: t2
        "
        )
    }

    /// Preserved child has a larger fetch → push our tighter limit.
    #[test]
    fn topk_pushed_when_child_has_larger_fetch() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // Child already has Sort(b ASC, fetch=10); our outer Sort has fetch=3 (tighter).
        let t1_with_sort = LogicalPlanBuilder::from(t1)
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(10))?
            .build()?;

        let plan = LogicalPlanBuilder::from(t1_with_sort)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(3))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=3
          Left Join: t1.a = t2.a
            Sort: t1.b ASC NULLS LAST, fetch=3
              TableScan: t1
            TableScan: t2
        "
        )
    }

    /// Preserved child already has a tighter fetch → skip pushdown.
    #[test]
    fn topk_not_pushed_when_child_has_smaller_fetch() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // Child already has Sort(b ASC, fetch=2); our outer Sort has fetch=5 (looser).
        let t1_with_sort = LogicalPlanBuilder::from(t1)
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(2))?
            .build()?;

        let plan = LogicalPlanBuilder::from(t1_with_sort)
            .join(
                LogicalPlanBuilder::from(t2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .sort_with_limit(vec![col("t1.b").sort(true, false)], Some(5))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Sort: t1.b ASC NULLS LAST, fetch=5
          Left Join: t1.a = t2.a
            Sort: t1.b ASC NULLS LAST, fetch=2
              TableScan: t1
            TableScan: t2
        "
        )
    }
}
