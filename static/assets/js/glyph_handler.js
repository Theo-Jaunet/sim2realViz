let scaleDist = d3.scaleLinear([0, 15], [0, 180]).clamp(true) // Hand craft value to normalize dist
const thick = 7
const rad = 6

function get_orr(p1, p2) {
    // return Math.atan2(p1[1] - p2[1]), (p1[0] - p2[0]) * (180 / Math.PI);
    let a = (Math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180) / Math.PI;
    // a -= 90;
    // a = a < 0 ? Math.abs(a) : 360 - a;

    a += 90;
    //   // a = a > 180 ? a - 180 : a + 180;

    //   a = (a + 180) % 360;
    //   a = a > 180 ? a - 180 : a + 180;

    return a;
}


function polarToCartesian(centerX, centerY, radius, angleInDegrees) {
    var angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;

    return {
        x: centerX + radius * Math.cos(angleInRadians),
        y: centerY + radius * Math.sin(angleInRadians)
    };
}


function describeArc(x, y, radius, startAngle, endAngle) {
    var start = polarToCartesian(x, y, radius, endAngle);
    var end = polarToCartesian(x, y, radius, startAngle);

    var largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";

    var d = [
        "M",
        start.x,
        start.y,
        "A",
        radius,
        radius,
        0,
        largeArcFlag,
        0,
        end.x,
        end.y
    ].join(" ");

    return d;
}


function drawInst(svg, data, id) {
    // Here data[0] == GT

    let g = svg.append("g").attr("class", "glyph").attr("num", id)

    let real_ang = get_orr(data[0], data[1]);


    const fix_gt = [mapSaleX.invert(data[0][0]), mapScaleY.invert(data[0][1])]
    const fix_real = [mapSaleX.invert(data[1][0]), mapScaleY.invert(data[1][1])]

    let real_dist = scaleDist(euclidian_dist(fix_gt, fix_real));

    // console.log(euclidian_dist(mapSaleX.invert(data[0]), mapScaleY.invert(data[1])));

    let real_st_ang = real_ang - real_dist / 2;
    let real_en_ang = real_ang + real_dist / 2;

    if (real_st_ang < 0) {
        real_st_ang = 360 - Math.abs(real_st_ang);
    }

    if (real_en_ang > 360) {
        real_en_ang = Math.abs(360 - real_en_ang);
    }


    let sim_ang = get_orr(data[0], data[2]);

    const fix_sim = [mapSaleX.invert(data[2][0]), mapScaleY.invert(data[2][1])]


    let sim_dist = scaleDist(euclidian_dist(fix_gt, fix_sim));

    let sim_st_ang = sim_ang - sim_dist / 2;
    let sim_en_ang = sim_ang + sim_dist / 2;

    if (sim_st_ang < 0) {
        sim_st_ang = 360 - Math.abs(sim_st_ang);
    }

    if (sim_en_ang > 360) {
        sim_en_ang = Math.abs(360 - sim_en_ang);
    }

    if (clip_bool)
        clipper.append("circle")
            .attr("class", "mapDottbrm")
            .attr("cx", data[0][0])
            .attr("cy", data[0][1])
            .attr("r", clip_rad)

    g.append('circle')
        // .attr("class", "slice")
        .attr("cx", (data[0][0]))
        .attr("cy", (data[0][1]))
        .attr("r", rad)
        .attr('fill', "none")
        .attr("stroke", "#555555")
        .style("stroke-width", "1.5px")
        .style("opacity", 0.25);

    g.append('path')
        .attr("class", "slice")
        .attr('d', describeArc((data[0][0]), (data[0][1]), rad, real_st_ang - 2, real_en_ang + 2))
        .attr('fill', "None")
        .attr("stroke", "#424242")
        .style("stroke-width", thick + 2 + "px")
        .style("opacity", 0.8);

    g.append('path')
        .attr("class", "slice")
        .attr('d', describeArc((data[0][0]), (data[0][1]), rad, real_st_ang, real_en_ang))
        .attr('fill', "None")
        .attr("stroke", glyph_realCol)
        .style("stroke-width", thick + "px")
        .style("opacity", 1);


    g.append('path')
        .attr("class", "slice")
        .attr('d', describeArc((data[0][0]), (data[0][1]), rad, sim_st_ang - 2, sim_en_ang + 2))
        .attr('fill', "None")
        .attr("stroke", "#424242")
        .style("stroke-width", thick + 2 + "px")
        .style("opacity", 0.8);


    g.append('path')
        .attr("class", "slice")
        .attr('d', describeArc((data[0][0]), (data[0][1]), rad, sim_st_ang, sim_en_ang))
        .attr('fill', "None")
        .attr("stroke", glyph_simCol)
        .style("stroke-width", thick + "px")
        .style("opacity", 1);


    // svg
    //   .append('rect')
    //   // .attr("class", "slice")
    //   // .attr('d', describeArc(100, 60, 50, 0, 150))
    //   .attr("x", 10)
    //   .attr("y", 10)
    //   .attr("width", 50)
    //   .attr("height", 50)
    //   .attr('fill', "None")
    //   .attr("stroke", "red")
    //   .style("stroke-width", "20px")
    //   .style("opacity", 0.7);


    if (contains2(sim_st_ang, sim_en_ang, real_st_ang)) {

        if (contains2(sim_st_ang, sim_en_ang, real_en_ang)) {
            svg
                .append('path')
                .attr("class", "slice")
                .attr('d', describeArc((data[0][0]), (data[0][1]), rad, real_st_ang, real_en_ang))
                .attr('fill', "None")
                .attr("stroke", glyph_overlap)
                .style("stroke-width", thick + "px")
                .style("opacity", 0.8);
        } else {
            svg
                .append('path')
                .attr("class", "slice")
                .attr('d', describeArc((data[0][0]), (data[0][1]), rad, real_st_ang, sim_en_ang))
                .attr('fill', "None")
                .attr("stroke", glyph_overlap)
                .style("stroke-width", thick + "px")
                .style("opacity", 0.8);
        }
    } else if (contains2(sim_st_ang, sim_en_ang, real_en_ang)) {
        svg
            .append('path')
            .attr("class", "slice")
            .attr('d', describeArc((data[0][0]), (data[0][1]), rad, sim_st_ang, real_en_ang))
            .attr('fill', "None")
            .attr("stroke", glyph_overlap)
            .style("stroke-width", thick + "px")
            .style("opacity", 0.8);
    } else if (contains2(real_st_ang, real_en_ang, sim_en_ang)) {
        svg
            .append('path')
            .attr("class", "slice")
            .attr('d', describeArc((data[0][0]), (data[0][1]), rad, sim_st_ang, sim_en_ang))
            .attr('fill', "None")
            .attr("stroke", glyph_overlap)
            .style("stroke-width", thick + "px")
            .style("opacity", 0.8);
    }
    d3.selectAll(".slice").raise()
}


function contains2(x0, x1, x) {
    if (x0 > 181 && x1 < 180) {
        x += 360;
        x1 += 360;
        return x >= x0 && x < x1;
    }
    // else if (x0 < 181 && x1 > 180) {
    //   x += 360;
    //   x0 += 360;
    //   return x >= x0 && x < x1;
    // }

    return x >= x0 && x < x1;
}
