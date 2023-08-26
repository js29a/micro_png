use micro_png::{read_png, write_apng, ImageData};

fn main() {
    // load an image
    let image = read_png("test.png").expect("can't load test.png");

    println!("{} x {}", image.width(), image.height());

    let data = image.data();

    (0 .. image.height()).for_each(|y| {
      (0 .. image.width()).for_each(|x| {
        let _pixel = data[y][x]; // (u16, u16, u16, u16)
      });
    });

    // now write it back as one-frame image

    write_apng("back.png",
        &ImageData::RGBA16(vec![data]),
        None ,// automatically select filtering
        None, // no progress callback
        false // no Adam-7
    ).expect("can't save back.png");
}
