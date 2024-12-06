use ndarray::{s, Array1, Array2, Array3};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn pad_times(scores: Vec<Vec<Vec<f32>>>, times: Vec<i32>) -> Vec<Vec<Vec<f32>>> {
    let mut scores_cp = Array3::from_shape_vec(
        (scores.len(), scores[0].len(), scores[0][0].len()),
        scores.into_iter().flatten().flatten().collect(),
    )
    .unwrap();

    for (i, &t) in times.iter().enumerate() {
        for j in t as usize..scores_cp.shape()[1] {
            for k in 0..scores_cp.shape()[2] {
                scores_cp[[i, j, k]] = f32::NEG_INFINITY;
                scores_cp[[i, k, j]] = f32::NEG_INFINITY;
            }
        }
    }

    scores_cp
        .outer_iter()
        .map(
            |mat: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>| {
                mat.outer_iter().map(|row| row.to_vec()).collect()
            },
        )
        .collect()
}

#[pyfunction]
fn prune_segments(
    segment_idxs: Vec<Vec<Vec<i32>>>,
    scores: Vec<Vec<f32>>,
    num_segments: usize,
) -> (Vec<Vec<Vec<i32>>>, Vec<Vec<f32>>) {
    let mut segment_idxs = Array3::from_shape_vec(
        (
            segment_idxs.len(),
            segment_idxs[0].len(),
            segment_idxs[0][0].len(),
        ),
        segment_idxs.into_iter().flatten().flatten().collect(),
    )
    .unwrap();
    let mut scores = Array2::from_shape_vec(
        (scores.len(), scores[0].len()),
        scores.into_iter().flatten().collect(),
    )
    .unwrap();

    let batch = segment_idxs.shape()[0];
    let seq_len = segment_idxs.shape()[1];

    for b in 0..batch {
        let mut selected_segments = Vec::new();
        let mut selected_scores = Vec::new();
        let mut remaining_segments: Vec<_> = (0..seq_len).collect();

        while selected_segments.len() < num_segments && !remaining_segments.is_empty() {
            let min_idx = remaining_segments
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| scores[[b, a]].partial_cmp(&scores[[b, b]]).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let min_segment = remaining_segments.remove(min_idx);
            selected_segments.push(segment_idxs.slice(s![b, min_segment, ..]).to_vec());
            selected_scores.push(scores[[b, min_segment]]);
        }

        segment_idxs.slice_mut(s![b, .., ..]).assign(
            &Array2::from_shape_vec(
                (selected_segments.len(), 2),
                selected_segments.into_iter().flatten().collect(),
            )
            .unwrap(),
        );

        scores
            .slice_mut(s![b, ..])
            .assign(&Array1::from_shape_vec(selected_scores.len(), selected_scores).unwrap());
    }

    (
        segment_idxs
            .outer_iter()
            .map(|mat| mat.outer_iter().map(|row| row.to_vec()).collect())
            .collect(),
        scores.outer_iter().map(|row| row.to_vec()).collect(),
    )
}

#[pymodule]
fn selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pad_times, m)?)?;
    m.add_function(wrap_pyfunction!(prune_segments, m)?)?; 
    
    Ok(())
}
