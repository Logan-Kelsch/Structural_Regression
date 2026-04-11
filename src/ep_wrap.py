from __future__ import annotations
import reproduction as _R
import initialization as _I
import evaluation as _E
import numpy as np
import sys



class Walker:

    def __init__(self, xi, xd, min_iters, exhaust=0.001):
        self.xi = xi
        self.xd = xd
        self.min_iters = min_iters
        self.exhaust = exhaust

        self.i_step = (xd - xi) / min_iters
        self.i_bped = -self.i_step / 2

        self.position = xi
        self.step = self.i_step
        self.bped = self.i_bped

    def walk(self, status: bool):
        if status:
            self.step *= 1.5
            self.bped /= 2

            # Clamp step so it cannot go above i_step
            self.step = min(self.step, self.i_step)

            self.position += self.step
        else:
            self.step /= 2
            self.bped *= 1.5

            # Clamp bped so it cannot go below i_bped
            self.bped = max(self.bped, self.i_bped)

            self.position += self.bped

        #print('step:',self.step,'bped:',self.bped)

        return self.position
    
    def is_exhausted(self):
        if((abs(self.step) + abs(self.bped)) <= self.exhaust):
            return True
        else:
            return False



from dataclasses import dataclass
import math
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt


@dataclass
class _Frame:
    frame_id: int
    parent_id: Optional[int]
    depth: int
    start: float
    end: float
    steps: int
    inner_steps: int
    milestones: List[float]
    index: int = 0

    def next_target(self) -> Optional[float]:
        if self.index >= len(self.milestones):
            return None
        return self.milestones[self.index]


class Logwalker:
    """
    1D recursive milestone walker.

    Root walk:
        start -> destination over `steps` milestones

    Milestones follow the halving-step finite-series construction:
        x_k = s + (d - s) * ((1 - 2^(-k)) / (1 - 2^(-steps)))
        for k = 1..steps

    Resolution:
    - Success at current milestone:
        move there, advance to next milestone
    - Failure at current milestone:
        create a nested child walk between:
            [last successful point, failed target]
        using current_frame.inner_steps milestones

    Nested step budget:
    - each frame stores:
        frame.steps
        frame.inner_steps = ceil(log_base(frame.steps)), clamped to >= 2
      where >= 2 means at least one intermediate step exists before the end target

    Exhaustion:
    - `exhaust_mode='steps'`: exhausted if len(path) >= exwhen
    - min_walk rule:
        after a new child frame is created, if that child's own `inner_steps`
        is less than `min_walk`, then exhaustion flips true immediately.

    Notes:
    - `path` stores realized position after each resolve()
    - `history` stores detailed event records
    - `frames_created` stores all generated frames for plotting/debugging
    """

    def __init__(
        self,
        start: float,
        destination: float,
        steps: int,
        exhaust_mode: str = "steps",
        exwhen: Optional[int] = None,
        log_base: float = 2.0,
        min_walk: int = 1,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if log_base <= 1:
            raise ValueError("log_base must be > 1")
        if min_walk < 1:
            raise ValueError("min_walk must be >= 1")
        if exwhen is not None and exwhen < 0:
            raise ValueError("exwhen must be >= 0 or None")

        self.start = float(start)
        self.destination = float(destination)
        self.steps = int(steps)

        self.exhaust_mode = exhaust_mode
        self.exwhen = math.inf if exwhen is None else int(exwhen)
        self.log_base = float(log_base)
        self.min_walk = int(min_walk)

        self.position = self.start
        self.best_position = self.start

        self.path: List[float] = []
        self.attempted: List[float] = []
        self.history: List[Dict[str, Any]] = []

        self._frame_counter = 0
        self.frames_created: List[Dict[str, Any]] = []

        self._stack: List[_Frame] = []
        self._min_walk_exhausted = False

        self._frame = self._make_frame(
            start=self.start,
            end=self.destination,
            steps=self.steps,
            depth=0,
            parent_id=None,
        )

        self.milestones = list(self._frame.milestones)

    def __repr__(self) -> str:
        return (
            f"Logwalker(position={self.position}, best_position={self.best_position}, "
            f"current_target={self.current_target()}, depth={self.depth}, "
            f"exhausted={self.is_exhausted()}, complete={self.is_complete()})"
        )

    @property
    def depth(self) -> int:
        return len(self._stack)

    @property
    def active_milestones(self) -> List[float]:
        return list(self._frame.milestones)

    def current_target(self) -> Optional[float]:
        return self._frame.next_target()

    def is_exhausted(self) -> bool:
        if self._min_walk_exhausted:
            return True
        if self.exhaust_mode == "steps":
            return len(self.path) >= self.exwhen
        raise NotImplementedError(f"Unsupported exhaust_mode: {self.exhaust_mode!r}")

    def is_complete(self) -> bool:
        return (not self._stack) and (self._frame.index >= len(self._frame.milestones))

    def resolve(self, passed: bool) -> float:
        """
        Resolve the current target with a boolean pass/fail.
        Returns the resulting position.
        """
        if self.is_exhausted() or self.is_complete():
            return self.position

        target = self.current_target()
        if target is None:
            return self.position

        event: Dict[str, Any] = {
            "passed": bool(passed),
            "attempted": target,
            "position_before": self.position,
            "best_before": self.best_position,
            "depth_before": self.depth,
            "frame_id": self._frame.frame_id,
            "frame_steps": self._frame.steps,
            "frame_inner_steps": self._frame.inner_steps,
            "frame_start": self._frame.start,
            "frame_end": self._frame.end,
            "frame_index_before": self._frame.index,
        }

        self.attempted.append(target)

        if passed:
            self.position = target
            self.best_position = target
            self._frame.index += 1
            event["action"] = "advance"

            # Bubble completed child successes back into parents.
            while self._frame.index >= len(self._frame.milestones) and self._stack:
                parent = self._stack.pop()

                # Parent is waiting at its failed milestone index.
                parent_target = parent.milestones[parent.index]

                self._frame = parent
                self.position = parent_target
                self.best_position = parent_target
                self._frame.index += 1

                event.setdefault("bubble_successes", []).append(
                    {
                        "into_parent_frame_id": self._frame.frame_id,
                        "accepted_parent_target": parent_target,
                    }
                )

        else:
            last_success = self._last_success_of_current_frame()
            failed_target = target

            child_steps = self._frame.inner_steps

            # Suspend current frame.
            parent_frame = self._frame
            self._stack.append(parent_frame)

            # Create child frame from last success -> failed target.
            self._frame = self._make_frame(
                start=last_success,
                end=failed_target,
                steps=child_steps,
                depth=parent_frame.depth + 1,
                parent_id=parent_frame.frame_id,
            )

            # On failure, position remains at last successful point.
            self.position = last_success
            self.best_position = last_success

            event["action"] = "refine"
            event["refine_from_frame_id"] = parent_frame.frame_id
            event["refine_to_frame_id"] = self._frame.frame_id
            event["refine_start"] = last_success
            event["refine_end"] = failed_target
            event["refine_steps"] = child_steps
            event["refine_inner_steps"] = self._frame.inner_steps
            event["refine_milestones"] = list(self._frame.milestones)

            # min_walk rule:
            # if this child would later attempt another nesting with too-small
            # inner_steps, flip exhaustion immediately.
            if self._frame.inner_steps < self.min_walk:
                self._min_walk_exhausted = True
                event["min_walk_exhausted"] = True
            else:
                event["min_walk_exhausted"] = False

        self.path.append(self.position)

        event["position_after"] = self.position
        event["best_after"] = self.best_position
        event["depth_after"] = self.depth
        event["frame_index_after"] = self._frame.index
        event["next_target"] = self.current_target()
        event["exhausted_after"] = self.is_exhausted()
        event["complete_after"] = self.is_complete()

        self.history.append(event)
        return self.position

    def step(self, passed: bool) -> float:
        """Alias for resolve()."""
        return self.resolve(passed)

    def reset(self) -> None:
        self.position = self.start
        self.best_position = self.start

        self.path.clear()
        self.attempted.clear()
        self.history.clear()
        self.frames_created.clear()

        self._frame_counter = 0
        self._stack.clear()
        self._min_walk_exhausted = False

        self._frame = self._make_frame(
            start=self.start,
            end=self.destination,
            steps=self.steps,
            depth=0,
            parent_id=None,
        )
        self.milestones = list(self._frame.milestones)

    def summary(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "destination": self.destination,
            "steps": self.steps,
            "position": self.position,
            "best_position": self.best_position,
            "root_milestones": list(self.milestones),
            "active_milestones": list(self.active_milestones),
            "current_target": self.current_target(),
            "path": list(self.path),
            "attempted": list(self.attempted),
            "n_steps_taken": len(self.path),
            "exhausted": self.is_exhausted(),
            "complete": self.is_complete(),
            "depth": self.depth,
            "min_walk": self.min_walk,
        }

    def plot_structure(self, figsize=(10, 5), show_labels: bool = True) -> None:
        """
        Plot all generated frames and their milestone positions.
        Deeper nesting levels are shown lower on the y-axis.
        """
        if not self.frames_created:
            print("No frame data to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        for rec in self.frames_created:
            depth = rec["depth"]
            y = -depth

            s = rec["start"]
            e = rec["end"]
            mids = rec["milestones"]

            ax.hlines(y=y, xmin=min(s, e), xmax=max(s, e), linewidth=2)
            ax.scatter(mids, [y] * len(mids), s=40)

            ax.scatter([s], [y], marker="|", s=250)
            ax.scatter([e], [y], marker="|", s=250)

            if show_labels:
                label = (
                    f"id={rec['frame_id']}  "
                    f"s={rec['steps']}  "
                    f"i={rec['inner_steps']}"
                )
                ax.text(
                    x=min(s, e),
                    y=y + 0.08,
                    s=label,
                    fontsize=8,
                    va="bottom",
                )

        ax.set_title("Logwalker frame structure")
        ax.set_xlabel("1D position")
        ax.set_ylabel("nest depth (negative for display)")
        ax.grid(True, alpha=0.25)
        plt.show()

    def plot_path(self, figsize=(10, 4), show_attempts: bool = True) -> None:
        """
        Plot realized walker position over decision index.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if self.path:
            x = list(range(1, len(self.path) + 1))
            ax.plot(x, self.path, marker="o", linewidth=1.5, label="realized position")

        if show_attempts and self.attempted:
            xa = list(range(1, len(self.attempted) + 1))
            ax.scatter(xa, self.attempted, s=30, alpha=0.8, label="attempted target")

        ax.axhline(self.start, linestyle="--", linewidth=1, label="start")
        ax.axhline(self.destination, linestyle="--", linewidth=1, label="destination")

        ax.set_title("Logwalker realized path")
        ax.set_xlabel("decision index")
        ax.set_ylabel("position")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.show()

    def _make_frame(
        self,
        start: float,
        end: float,
        steps: int,
        depth: int,
        parent_id: Optional[int],
    ) -> _Frame:
        frame_id = self._frame_counter
        self._frame_counter += 1

        inner_steps = self._compute_inner_steps(steps)
        milestones = self._generate_milestones(start, end, steps)

        frame = _Frame(
            frame_id=frame_id,
            parent_id=parent_id,
            depth=depth,
            start=float(start),
            end=float(end),
            steps=int(steps),
            inner_steps=int(inner_steps),
            milestones=milestones,
            index=0,
        )

        self.frames_created.append(
            {
                "frame_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "depth": frame.depth,
                "start": frame.start,
                "end": frame.end,
                "steps": frame.steps,
                "inner_steps": frame.inner_steps,
                "milestones": list(frame.milestones),
            }
        )

        return frame

    def _last_success_of_current_frame(self) -> float:
        if self._frame.index == 0:
            return self._frame.start
        return self._frame.milestones[self._frame.index - 1]

    def _compute_inner_steps(self, steps: int) -> int:
        """
        Compute the next nesting budget.

        Minimum of 2 milestones means:
        - one intermediate step
        - one final endpoint
        """
        raw = math.log(max(steps, 2), self.log_base)
        return max(2, int(math.ceil(raw)))

    @staticmethod
    def _generate_milestones(start: float, destination: float, steps: int) -> List[float]:
        """
        x_k = s + (d - s) * ((1 - 2^(-k)) / (1 - 2^(-steps))), k=1..steps
        """
        start = float(start)
        destination = float(destination)
        steps = int(steps)

        if steps < 1:
            raise ValueError("steps must be >= 1")

        if steps == 1:
            return [destination]

        denom = 1.0 - 2.0 ** (-steps)
        return [
            start + (destination - start) * ((1.0 - 2.0 ** (-k)) / denom)
            for k in range(1, steps + 1)
        ]


def evolve_population(
    X = None, 
    G = None, 
    iterations = 0,
    early_stop = 0.0,
    break_extinction = False,
    initialization_kwargs = None,
    solver_kwargs = None,
    selector_kwargs = None,
    chunk_num = None,
    return_stats    :   bool    =   False
):
    '''returns X, G, final evaluation (and optionally stats)'''

    #-----------------------------------------------
    # KWARGS FORMATTING FOR NO PARAMETER DEFINITION
    # AND GRAMMAR AND POPULATION FIRST GENERATIONS
    #-----------------------------------------------

    if(initialization_kwargs is None):
        initialization_kwargs = {
            "structure"   :   'Intraday',
            "incl_time"   :   True,
            "data_file"   :   '../data/spy5m.csv',
            "epoch_idx"   :   [0],
            "hlocv_idx"   :   [1,2,3,4],
            "pop_size"    :   150,
            "grmr_type"   :   'Null',
            "grmr_mdl"    :   240,
            "chunk_size"  :   25,
            "wf_windows"  :   2,
            "verbose"     :   0
        }

    # X and G should come in to this function
    #if we want to iterate from a given population
    if(X is None or G is None):
        print('No Grammar OR Population provided.\n'
              'Initializing new population.')
        
        #we will initialize a new population and grammar if we got nothing
        X, G = _I.initialize(**initialization_kwargs)

    if(solver_kwargs is None):
        solver_kwargs = {
            "offset"    :   5,
            "t_vec"		:	'Close',
            "t_mode"	:	'AD',
            "emission"	:	[
                {"ID": 5, "alpha": 
                    {"ID": 3, "x": "tvec", "delta1": 20, "offset": False}},
                {"ID": "divide"},
                {"ID": 18, "x": "tvec", "delta1": 20, "offset": False},
            ],
            "AD_cond"   :	('gt', 2),
        }

    if(selector_kwargs is None):
        selector_kwargs = {
            "method"    :   "Threshold",
            "percent"   :   0.2
        }
    
    #initialize solver and selector
    solver  = _E.Solver(X, **solver_kwargs)
    selector= _R.Selector(**selector_kwargs) 

    #the loop really should count reproduction iterations,
    #we need an evaluation to reproduce and will want to end
    #with a final evaluation, so we will loop E, R and end E

    #list variables for holding stats collected if user wants them returned
    if(return_stats):
        instantiation_stats_stack = []
        reproduction_stats_stack  = []

    for i in range(iterations):
        evaluation, inst_stats = _E.evaluate(X, solver, chunk_num)

        qgenes = np.count_nonzero(evaluation["F"]>0)

        if(i==0):
            #initial print of solution stats
            ftcount = np.unique_counts(evaluation['svecs']['anomaly_mask'])[1]
            #print()

        if(break_extinction and qgenes==0):
            #print('All genes failed, breaking loop.')
            break

        sys.stdout.write("\r\033[2K")
        if(qgenes<10):
            sys.stdout.write(f"P:{100*(ftcount[1]/(ftcount[0]+ftcount[1])):.2f}% Gen {i} |" + "_" * min(qgenes, 29) + str(qgenes) + "_" * (29 - min(qgenes, 29)) + "| ")
        else:
            sys.stdout.write(f"P:{100*(ftcount[1]/(ftcount[0]+ftcount[1])):.2f}% Gen {i} |" + "_" * min(qgenes, 28) + str(qgenes) + "_" * (28 - min(qgenes, 28)) + "| ")
        sys.stdout.flush()
        #print(f'Gen {i}: {qgenes} Quality Genes. ', end='')

        print("1" if True else "2")

        
        #print('Reproducting...')

        reproduction_stats = _R.reproduce(X,G, selector, evaluation)

        
        

        #add on stats if user wants them returned
        if(return_stats):
            instantiation_stats_stack.append(inst_stats)
            reproduction_stats_stack.append(reproduction_stats)

        #break case for enough success to end iteration
        if(early_stop > 0 and (qgenes / X._max_size) > early_stop):
            #print(f'Success Reached ({early_stop:.2f}) in Population. Breaking evolution loop.')
            break



    #final gene evaluation
    evaluation, inst_stats = _E.evaluate(X, solver, chunk_num)

    #add on last stats if the user wants them returned
    if(return_stats):
        instantiation_stats_stack.append(inst_stats)
        stats = {
            "Instantiation":instantiation_stats_stack,
            "Reproduction":reproduction_stats_stack         
        }

        #then if the user wants the stats returned we should probably also return the stats
        return X, G, evaluation, stats
    else:
        return X, G, evaluation
    


from importlib import reload
import ep_wrap as ep
import evaluation as _E
import visualization as _V
import initialization as _I
import reproduction as _R
import numpy as np
import utility as _util
import matplotlib.pyplot as plt
from copy import deepcopy




def solver_inner(
    initialization_kwargs,
    solver_kwargs,
    logwalker_kwargs,
    G = None,
    chunk_num = 0,
    purge_thresh = 0.05,
):
    walker = Logwalker(**logwalker_kwargs)

    i = 0

    X, G, evaluation = ep.evolve_population(
        G=G,
        iterations=10,
        early_stop=0.1,
        initialization_kwargs=initialization_kwargs,
        solver_kwargs=solver_kwargs,
        chunk_num=chunk_num
    )
    print()

    _R.purge_indistinguishable(X, evaluation, threshold=purge_thresh, chunk_num=chunk_num)

    # dynamic walker-controlled threshold term
    solver_kwargs["emission"].append({"ID": 5, "alpha": 0.0})

    while True:

        # stop checks belong at top, before trying to set/evaluate a new target
        if walker.is_complete():
            print("Logwalker reached destination.")
            break

        if walker.is_exhausted():
            print("Logwalker has been exhausted.")
            break

        target = walker.current_target()
        solver_kwargs["emission"][-1]["alpha"] = target

        # keep previous state so failed attempts can be discarded
        X_prev = deepcopy(X)
        G_prev = deepcopy(G)

        X_try, G_try, evaluation = ep.evolve_population(
            X, G,
            iterations=15,
            early_stop=0.25,
            break_extinction=True,
            initialization_kwargs=initialization_kwargs,
            solver_kwargs=solver_kwargs,
            chunk_num=chunk_num
        )

        success = _util.is_successful_evolution(evaluation)

        # resolve the walker only after success/failure is known for the current target
        walker.step(success)

        if success:
            X, G = deepcopy(X_try), deepcopy(G_try)
            s_idx = np.where(evaluation["F"] > 0)[0]
            #does not clean population if it will end up breaking this loop
            if(not walker.is_complete() and not walker.is_exhausted()):
                s_idx = _R.purge_indistinguishable(X, evaluation, threshold=purge_thresh, chunk_num=chunk_num)
            i += 1
            print(
                f"SUCCESS | @ {walker.position:.4f} "
                f"-> {walker.current_target():.4f}"
            )
        else:
            X, G = deepcopy(X_prev), deepcopy(G_prev)
            i -= 1
            #print(
            #    f"FAILURE | @ {walker.position:.4f} "
            #    f"-> {walker.current_target():.4f}"
            #)
        

        # stop checks again after the walker has updated
        if walker.is_complete():
            print("Logwalker reached destination.")
            break

        if walker.is_exhausted():
            print("Logwalker has been exhausted.")
            break
    
    return X, G, s_idx, walker, evaluation