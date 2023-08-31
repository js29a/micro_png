use micro_png::*;

fn from_readme() {
    // load an image
    let image = read_png("fixtures/test.png").expect("can't load test.png");

    println!("{} x {}", image.width(), image.height());

    let data = image.data();

    (0 .. image.height()).for_each(|y| {
      (0 .. image.width()).for_each(|x| {
        let _pixel = data[y][x]; // (u16, u16, u16, u16)
      });
    });

    // now write it back as one-frame image

    write_apng("tmp/back.png",
        ImageData::RGBA16(vec![data]),
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
        ImageData::RGBA(vec![data]), // write one frame
        None ,// automatically select filtering
        None, // no progress callback
        false // no Adam-7
    ).expect("can't save back.png");

    // write single-framed image usize builder

    let data_1: Vec<Vec<RGBA>> = vec![
        vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
        vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
    ];

    let builder = APNGBuilder::new("tmp/foo.png", ImageData::RGBA(vec![data_1]))
        .set_adam_7(true);

    build_apng(builder).unwrap();

   // write some animations

    let data_2 = vec![
        vec![// frame #0
            vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
            vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
        ],
        vec![// frame #1
            vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 1st line
            vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 2nd line
        ],
        vec![// frame #2
            vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 1st line
            vec![(255, 255, 0, 255), (0, 255, 0, 255)],// the 2nd line
        ],
    ];

    build_apng(
       APNGBuilder::new("tmp/bar.png", ImageData::RGBA(data_2))
           .set_def_dur((100, 1000)) // default frame duration: 100 / 1000 [sec]
           .set_dur(1, (500, 1000)) // duration for frame #1: 500 / 1000 [sec]
    ).unwrap();
}

fn from_wiki_copy() {
    // 1. load the file
    let image = read_png("fixtures/test.png").expect("can't load test.png");

    // 2. save it as TrueColor RGBA
    if let ImageData::RGBA(rgba) = image.raw() {
        build_apng(APNGBuilder::new("tmp/test-copy.png", ImageData::RGBA(rgba.to_vec()))).unwrap()
    }
    else {
        panic!("sth wrong with fixtures/test.png")
    }

    // 3. save it as HDR (16 bit component depth)
    build_apng(APNGBuilder::new("tmp/test-HDR.png", ImageData::RGBA16(vec![image.data()]))).unwrap();
}

fn main() {
    from_readme();
    from_wiki_copy();
}
