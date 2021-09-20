let fake_xrange = [-1.6, 18.93];
let fake_yrange = [-1.3, 19.81];
let fake_rrange = [0, 360];

function getRandomFloat(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.random() * (max - min + 1) + min;
}


fake_models(3);

function fake_models(nmod, ndots) {

    return d3.range(nmod).map(d => {
        return {
            "sim_dots": get_dots(ndots),
            "real_dots": get_dots(ndots * 0.1),
            "sim_perf": Math.random(),
            "real_perf": Math.random()
        }
    })


}


function get_dots(n) {
    return d3.range(n).map(d => {
        return {
            "x": getRandomFloat(fake_xrange[0], fake_xrange[1]),
            "y": getRandomFloat(fake_yrange[0], fake_yrange[1]),
            "r": getRandomFloat(fake_rrange[0], fake_rrange[1]),
            "gt_x": getRandomFloat(fake_xrange[0], fake_xrange[1]),
            "gt_y": getRandomFloat(fake_yrange[0], fake_yrange[1]),
            "gt_r": getRandomFloat(fake_rrange[0], fake_rrange[1])

        }
    })

}