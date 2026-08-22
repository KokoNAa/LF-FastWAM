from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        hard_curriculum_groups = getattr(
            self.dataset, "pgc_v9_hard_curriculum_group_ids", None
        )
        closed_loop_curriculum_groups = getattr(
            self.dataset, "pgc_v9_closed_loop_group_ids", None
        )
        if (
            hard_curriculum_groups is not None
            and closed_loop_curriculum_groups is not None
        ):
            raise ValueError(
                "PGC hard-role and closed-loop curricula are mutually exclusive."
            )
        curriculum_groups = (
            closed_loop_curriculum_groups
            if closed_loop_curriculum_groups is not None
            else hard_curriculum_groups
        )
        curriculum_name = (
            "PGC V9.12 closed-loop"
            if closed_loop_curriculum_groups is not None
            else "PGC V9.6 hard-role"
        )
        if curriculum_groups is None:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            groups = [int(value) for value in curriculum_groups]
            if len(groups) != len(self.dataset):
                raise ValueError(
                    f"{curriculum_name} curriculum labels must match dataset length."
                )
            unique_groups = sorted(set(groups))
            if unique_groups != [0, 1, 2, 3]:
                raise ValueError(
                    f"{curriculum_name} curriculum requires exactly four groups."
                )
            global_window = (
                self.batch_size
                * self.num_processes
                * self.gradient_accumulation_steps
            )
            if global_window % 4 or self.gradient_accumulation_steps % 4:
                raise ValueError(
                    f"{curriculum_name} requires gradient accumulation divisible by 4 "
                    "so every rank and global optimizer window sees all four "
                    "curriculum groups."
                )
            grouped_positions = {
                group: [
                    index for index, value in enumerate(groups) if value == group
                ]
                for group in unique_groups
            }
            for group, positions in grouped_positions.items():
                if not positions:
                    raise ValueError(
                        f"{curriculum_name} curriculum group {group} is empty."
                    )
                order = torch.randperm(len(positions), generator=g).tolist()
                grouped_positions[group] = [positions[index] for index in order]

            # The dataset builder gives every group equal cardinality. Build a
            # rank-aware optimizer-window schedule: group=(microstep+rank+
            # local_batch_slot)%4. Thus every rank sees all four groups over
            # four microsteps, and their global aggregate is also exactly 1:1.
            group_size = len(grouped_positions[0])
            if any(len(values) != group_size for values in grouped_positions.values()):
                raise ValueError(
                    f"{curriculum_name} curriculum groups must have equal cardinality."
                )
            window_groups = [
                (microstep + process + batch_slot) % 4
                for microstep in range(self.gradient_accumulation_steps)
                for process in range(self.num_processes)
                for batch_slot in range(self.batch_size)
            ]
            per_group_per_window = global_window // 4
            if any(
                window_groups.count(group) != per_group_per_window
                for group in range(4)
            ):
                raise RuntimeError(
                    f"{curriculum_name} rank-aware curriculum is imbalanced."
                )
            complete_windows, remainder = divmod(
                group_size, per_group_per_window
            )
            scheduled_groups = window_groups * complete_windows
            scheduled_groups.extend(
                group
                for _ in range(remainder)
                for group in range(4)
            )
            cursors = [0, 0, 0, 0]
            indices = []
            for group in scheduled_groups:
                indices.append(grouped_positions[group][cursors[group]])
                cursors[group] += 1
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
