use micro_png::{read_png, write_apng, ImageData, RGBA};

fn main() {
    // load an image
    let image = read_png("tmp/test.png").expect("can't load test.png");

    println!("{} x {}", image.width(), image.height());

    let data = image.data();

    (0 .. image.height()).for_each(|y| {
      (0 .. image.width()).for_each(|x| {
        let _pixel = data[y][x]; // (u16, u16, u16, u16)
      });
    });

    // now write it back as one-frame image

    write_apng("tmp/back.png",
        &ImageData::RGBA16(vec![data]),
        None ,// automatically select filtering
        None, // no progress callback
        false // no Adam-7
    ).expect("can't save back.png");

    // write 2x2 pattern image

    let data: Vec<Vec<RGBA>> = vec![
        vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
        vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
    ];

    write_apng("tmp/2x2.png",
        &ImageData::RGBA(vec![data]), // write one frame
        None ,// automatically select filtering
        None, // no progress callback
        false // no Adam-7
    ).expect("can't save back.png");
}
