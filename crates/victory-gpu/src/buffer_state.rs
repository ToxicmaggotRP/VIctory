//! Pure state tracking for D3D12 host copies of title buffers.
//!
//! Buffers can be used as UAVs or in several read-only roles. D3D12 permits
//! those read-only roles to be combined into one state; this module keeps
//! the state decisions independent of the Windows API so they can be tested
//! on every host.

pub(crate) const UNORDERED_ACCESS: u32 = 0x8;
pub(crate) const GENERIC_READ: u32 = 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Barrier {
    Transition { before: u32, after: u32 },
    Uav,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BufferState {
    first: u32,
    last: u32,
}

impl BufferState {
    /// A list's first and current access state. The first access needs no
    /// in-list barrier: the submission prelude moves the real resource there.
    pub(crate) fn new(wanted: u32) -> Self {
        let state = normalize(wanted);
        Self {
            first: state,
            last: state,
        }
    }

    /// Records another access in the same command list.
    ///
    /// A transition orders accesses when the state changes. Repeated UAV
    /// accesses need an explicit UAV barrier because the resource stays in
    /// the same state while writes may depend on earlier writes.
    pub(crate) fn record(&mut self, wanted: u32) -> Option<Barrier> {
        let next = normalize(wanted);
        let before = std::mem::replace(&mut self.last, next);
        if before != next {
            Some(Barrier::Transition {
                before,
                after: next,
            })
        } else if next == UNORDERED_ACCESS {
            Some(Barrier::Uav)
        } else {
            None
        }
    }

    /// Barrier required before this list starts, given the state left by the
    /// previous submission/list.
    ///
    /// Even if both lists use UAV, their command-list boundary is not used as
    /// a substitute for a UAV memory barrier.
    pub(crate) fn prelude(&self, actual: u32) -> Option<Barrier> {
        if actual != self.first {
            Some(Barrier::Transition {
                before: actual,
                after: self.first,
            })
        } else if self.first == UNORDERED_ACCESS {
            Some(Barrier::Uav)
        } else {
            None
        }
    }

    /// State left by this list, to be carried into the next submission.
    pub(crate) fn submitted_state(&self) -> u32 {
        self.last
    }
}

fn normalize(wanted: u32) -> u32 {
    debug_assert!(
        wanted == UNORDERED_ACCESS || wanted & !GENERIC_READ == 0,
        "unsupported buffer resource state {wanted:#x}"
    );
    if wanted == UNORDERED_ACCESS {
        UNORDERED_ACCESS
    } else {
        GENERIC_READ
    }
}

#[cfg(test)]
mod tests {
    use super::{Barrier, BufferState, GENERIC_READ, UNORDERED_ACCESS};

    const COMMON: u32 = 0;
    const VERTEX: u32 = 0x1;
    const INDEX: u32 = 0x2;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const INDIRECT_ARGUMENT: u32 = 0x200;

    #[test]
    fn read_only_accesses_share_generic_read_state() {
        for access in [
            VERTEX,
            INDEX,
            SHADER_RESOURCE,
            INDIRECT_ARGUMENT,
            GENERIC_READ,
        ] {
            let state = BufferState::new(access);
            assert_eq!(state.submitted_state(), GENERIC_READ, "access {access:#x}");
        }

        assert_eq!(
            BufferState::new(UNORDERED_ACCESS).submitted_state(),
            UNORDERED_ACCESS
        );
    }

    #[test]
    fn transitions_cover_uav_read_and_back() {
        let mut state = BufferState::new(UNORDERED_ACCESS);

        assert_eq!(
            state.record(VERTEX),
            Some(Barrier::Transition {
                before: UNORDERED_ACCESS,
                after: GENERIC_READ,
            })
        );
        assert_eq!(state.record(INDIRECT_ARGUMENT), None);
        assert_eq!(
            state.record(UNORDERED_ACCESS),
            Some(Barrier::Transition {
                before: GENERIC_READ,
                after: UNORDERED_ACCESS,
            })
        );
        assert_eq!(state.submitted_state(), UNORDERED_ACCESS);
    }

    #[test]
    fn repeated_uav_accesses_require_uav_barriers() {
        let mut state = BufferState::new(UNORDERED_ACCESS);
        assert_eq!(state.record(UNORDERED_ACCESS), Some(Barrier::Uav));
        assert_eq!(state.record(UNORDERED_ACCESS), Some(Barrier::Uav));
    }

    #[test]
    fn read_after_read_does_not_emit_a_barrier() {
        let mut state = BufferState::new(VERTEX);
        assert_eq!(state.record(INDEX), None);
        assert_eq!(state.record(SHADER_RESOURCE), None);
        assert_eq!(state.record(INDIRECT_ARGUMENT), None);
    }

    #[test]
    fn prelude_transitions_from_the_previous_list_state() {
        let first = BufferState::new(UNORDERED_ACCESS);
        assert_eq!(
            first.prelude(COMMON),
            Some(Barrier::Transition {
                before: COMMON,
                after: UNORDERED_ACCESS,
            })
        );

        let second = BufferState::new(VERTEX);
        assert_eq!(
            second.prelude(first.submitted_state()),
            Some(Barrier::Transition {
                before: UNORDERED_ACCESS,
                after: GENERIC_READ,
            })
        );
    }

    #[test]
    fn consecutive_uav_lists_get_a_prelude_barrier() {
        let first = BufferState::new(UNORDERED_ACCESS);
        let second = BufferState::new(UNORDERED_ACCESS);

        assert_eq!(second.prelude(first.submitted_state()), Some(Barrier::Uav));
    }

    #[test]
    fn matching_read_only_list_states_need_no_prelude() {
        let first = BufferState::new(VERTEX);
        let second = BufferState::new(INDIRECT_ARGUMENT);

        assert_eq!(second.prelude(first.submitted_state()), None);
    }
}
