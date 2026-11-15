/** Help for the Trajectory panel: where pseudotime starts, and genes along a branch. */
export const TRAJECTORY_HELP: Record<string, string> = {
  "trajectory.root-kind":
    "Where pseudotime starts. A group starts from that group's most central cell, so the whole group sits at the start. One cell starts from exactly that cell: use it when the start state is the tip of a group, or no group matches it. Type the cell's name or pick it on the plot.",
  "trajectory.rank-by":
    "How genes are ranked on a branch. Changing along pseudotime: the strongest trends on the branch, which often repeat the early genes every branch shares. Higher on this branch: genes whose curve sits above the other branches' at the same pseudotime, so the list names this branch's markers.",
};
