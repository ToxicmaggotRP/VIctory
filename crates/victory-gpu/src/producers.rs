//! Diagnostics: draws and dispatches that read what only dropped work writes.
//!
//! A draw or dispatch the host cannot run (its shader does not translate, its
//! pipeline is refused or skipped) leaves what it would have written as the
//! title last saw it. A later pass that reads that, or waits for a value in
//! it, works on stale data or never finishes, which on some hosts is a
//! device hang far from the dropped work. This module keeps what dropped work
//! would have written (its unordered-access views and render targets), forgets
//! a range again when work that runs writes it, and logs once per pair
//! a reader that runs reading one of them, with the shader that was dropped.

use std::collections::HashSet;
use std::sync::Mutex;

use crate::descriptor::View;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    /// Title memory, `start..end`.
    Buffer(u64, u64),
    /// A title texture object.
    Texture(usize),
}

impl Target {
    fn overlaps(self, other: Target) -> bool {
        match (self, other) {
            (Target::Buffer(a, b), Target::Buffer(c, d)) => a < d && c < b,
            (Target::Texture(a), Target::Texture(b)) => a == b,
            _ => false,
        }
    }

    fn describe(self) -> String {
        match self {
            Target::Buffer(a, b) => format!("buffer {a:#x}+{:#x}", b - a),
            Target::Texture(t) => {
                let desc = crate::resource::host_resource(t as crate::object::This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!(" {}x{} f{}", d.Width, d.Height, d.Format.0)
                });
                format!("texture {t:#x}{}", desc.unwrap_or_default())
            }
        }
    }
}

struct Write {
    what: Target,
    writer: String,
    frame: u64,
}

static WRITES: Mutex<Vec<Write>> = Mutex::new(Vec::new());
static SEEN: Mutex<Option<HashSet<(String, String)>>> = Mutex::new(None);

thread_local! {
    /// The pipeline the draw or dispatch being bound on this thread runs.
    static READER: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Whether the detector runs (`VICTORY_PRODUCERS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRODUCERS").is_some_and(|v| v == "1"))
}

/// The pipeline a later [`read`] or [`live_write`] on this thread belongs to.
pub(crate) fn set_reader(name: &str) {
    if on() {
        READER.with(|r| {
            let mut r = r.borrow_mut();
            r.clear();
            r.push_str(name);
        });
    }
}

fn target_of(view: &View) -> Option<Target> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
    {
        return Some(Target::Buffer(address, address + len));
    }
    match *view {
        View::Shader(r, _) | View::Unordered(r, _, _) => Some(Target::Texture(r)),
        _ => None,
    }
}

fn record(what: Target, writer: &str) {
    let frame = crate::queue::presented();
    let mut writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    // What the frame before last dropped is history: the title has moved on.
    writes.retain(|w| w.frame + 2 > frame);
    if writes.len() >= 8192 {
        writes.remove(0);
    }
    if !writes.iter().any(|w| w.what == what && w.writer == writer) {
        writes.push(Write {
            what,
            writer: writer.to_owned(),
            frame,
        });
    }
}

/// Dropped work `writer` would have written through `view`.
pub(crate) fn dropped_write(writer: &str, view: &View) {
    if on()
        && let Some(what) = target_of(view)
    {
        record(what, writer);
    }
}

/// Dropped draw `writer` would have rendered to title texture `resource`.
pub(crate) fn dropped_target(writer: &str, resource: usize) {
    if on() && resource != 0 {
        record(Target::Texture(resource), writer);
    }
}

/// Work that runs writes through `view`: only the bytes it actually writes
/// are no longer attributed to a dropped producer. Any untouched part of an
/// earlier dropped range remains a possible stale read.
pub(crate) fn live_write(view: &View) {
    if !on() {
        return;
    }
    let Some(written) = target_of(view) else {
        return;
    };

    const MAX_WRITES: usize = 8192;
    let mut writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::mem::take(&mut *writes);
    let mut remaining = Vec::with_capacity(previous.len());

    for write in previous {
        for what in unwritten_parts(write.what, written) {
            remaining.push(Write {
                what,
                writer: write.writer.clone(),
                frame: write.frame,
            });
        }
    }

    // Range splitting can create more entries than it consumes. Keep the
    // detector bounded just as record() does, dropping the oldest entries.
    if remaining.len() > MAX_WRITES {
        remaining.drain(..remaining.len() - MAX_WRITES);
    }
    *writes = remaining;
}

/// The portions of `target` not overwritten by `written`.
///
/// Buffer ranges are half-open. A write in the middle leaves both the prefix
/// and suffix attributed to the dropped producer; a write covering the whole
/// range removes it. Textures are tracked as indivisible resources.
fn unwritten_parts(target: Target, written: Target) -> Vec<Target> {
    match (target, written) {
        (Target::Buffer(start, end), Target::Buffer(write_start, write_end)) => {
            let overlap_start = start.max(write_start);
            let overlap_end = end.min(write_end);
            if overlap_start >= overlap_end {
                return vec![target];
            }

            let mut remaining = Vec::with_capacity(2);
            if start < overlap_start {
                remaining.push(Target::Buffer(start, overlap_start));
            }
            if overlap_end < end {
                remaining.push(Target::Buffer(overlap_end, end));
            }
            remaining
        }
        (Target::Texture(a), Target::Texture(b)) if a == b => Vec::new(),
        _ => vec![target],
    }
}

/// A draw that runs renders to title texture `resource`.
pub(crate) fn live_target(resource: usize) {
    if on() && resource != 0 {
        WRITES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|w| w.what != Target::Texture(resource));
    }
}

/// Work that runs reads through `view`: logs it once per (reader, dropped
/// writer) if only dropped work wrote that.
pub(crate) fn read(view: &View) {
    if !on() {
        return;
    }
    let Some(what) = target_of(view) else { return };
    let hits: Vec<(String, Target)> = WRITES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|w| w.what.overlaps(what))
        .map(|w| (w.writer.clone(), w.what))
        .collect();
    if hits.is_empty() {
        return;
    }
    let reader = READER.with(|r| r.borrow().clone());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(HashSet::new);
    for (writer, written) in hits {
        if seen.insert((reader.clone(), writer.clone())) {
            tracing::warn!(
                reader = %reader,
                dropped = %writer,
                what = %written.describe(),
                frame = crate::queue::presented(),
                "reads what only dropped work writes"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Target, unwritten_parts};

    #[test]
    fn partial_buffer_write_preserves_both_unwritten_sides() {
        assert_eq!(
            unwritten_parts(Target::Buffer(0x1000, 0x2000), Target::Buffer(0x1400, 0x1800)),
            vec![
                Target::Buffer(0x1000, 0x1400),
                Target::Buffer(0x1800, 0x2000),
            ],
        );
    }

    #[test]
    fn buffer_write_trims_a_prefix_or_suffix() {
        assert_eq!(
            unwritten_parts(Target::Buffer(0x1000, 0x2000), Target::Buffer(0x0800, 0x1400)),
            vec![Target::Buffer(0x1400, 0x2000)],
        );
        assert_eq!(
            unwritten_parts(Target::Buffer(0x1000, 0x2000), Target::Buffer(0x1800, 0x2800)),
            vec![Target::Buffer(0x1000, 0x1800)],
        );
    }

    #[test]
    fn covering_buffer_write_removes_the_stale_range() {
        assert!(
            unwritten_parts(Target::Buffer(0x1000, 0x2000), Target::Buffer(0x0800, 0x2800))
                .is_empty()
        );
    }

    #[test]
    fn disjoint_and_adjacent_buffer_writes_leave_the_range_unchanged() {
        let target = Target::Buffer(0x1000, 0x2000);
        assert_eq!(
            unwritten_parts(target, Target::Buffer(0x2000, 0x3000)),
            vec![target],
        );
        assert_eq!(
            unwritten_parts(target, Target::Buffer(0x0800, 0x1000)),
            vec![target],
        );
        assert_eq!(
            unwritten_parts(target, Target::Buffer(0x3000, 0x4000)),
            vec![target],
        );
    }

    #[test]
    fn textures_are_removed_only_by_a_write_to_the_same_texture() {
        assert!(
            unwritten_parts(Target::Texture(7), Target::Texture(7)).is_empty()
        );
        assert_eq!(
            unwritten_parts(Target::Texture(7), Target::Texture(8)),
            vec![Target::Texture(7)],
        );
    }

    #[test]
    fn buffer_and_texture_targets_do_not_overlap() {
        let buffer = Target::Buffer(0x1000, 0x2000);
        assert_eq!(
            unwritten_parts(buffer, Target::Texture(7)),
            vec![buffer],
        );
        assert_eq!(
            unwritten_parts(Target::Texture(7), buffer),
            vec![Target::Texture(7)],
        );
    }
}
