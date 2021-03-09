use plotters::prelude::*;
use std::cmp::max;

pub(crate) struct Visualize {}

impl Visualize {
    pub fn plot(training_loss: Vec<f64>, validation_loss: Vec<f64>) {
        let max_value = training_loss
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let root_area = BitMapBackend::new("plots/training_and_validation_loss.png", (600, 400))
            .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("Training and Validation Loss", ("sans-serif", 40))
            .build_cartesian_2d(0.0..training_loss.len() as f64, 0.0..max_value.clone())
            .unwrap();

        ctx.configure_mesh().draw().unwrap();

        ctx.draw_series(LineSeries::new(
            training_loss
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, y.clone())),
            &BLUE,
        ))
        .unwrap()
        .label("Training loss");
        ctx.draw_series(LineSeries::new(
            validation_loss
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, y.clone())),
            &RED,
        ))
        .unwrap()
        .label("Validation loss");
    }
}
