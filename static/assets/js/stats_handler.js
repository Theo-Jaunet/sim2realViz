let n_portions = 72;
let stats_r = 65
let stats_orr_scale = d3.scaleLinear().domain([0, 20]).range([0, stats_r]).clamp(true)

let roomLabels = ["middle", "hall", "2", "Left", "4", "5", "6", "7", "8", "9", "10", "11", "12", "back", "right"];

let roomsBounds = [
    [[[7.3, 2.6], [12.55, 15.9]]],
    [[[6.4, -1.15], [14.15, 2.63]]],
    [[[-1.5, 2.56], [5.68, 5.75]], [[-1.5, -0.9], [1.6, 3.49]]],
    [[[12.5, 2.1], [14, 16.1]]],
    [[[-1.5, 5.9], [5.74, 11.04]]],
    [[[-1.5, 11.2], [3.9, 16.4]], [[2.3, 11.54], [5.75, 13]]],
    [[[14, 16], [18.92, 19.79]]],
    [[[14, 12.7], [19, 16.04]]],
    [[[14, 9.2], [19, 12.55]]],
    [[[14, 5.8], [19, 9.14]]],
    [[[14, 2.3], [19, 5.6]]],
    [[[14, -1.5], [19, 2.06]]],
    [[[1.9, 0.54], [5.8, 2.35]]],
    [[[3.7, 16.01], [14.1, 18.04]], [[3.97, 13.02], [7.3, 18.04]]],
    [[[5.7, 2.40], [7.3, 13.24]]]
];


let id_stock = {
    "sim": [],
    "real": [],
    "gt": []
}

let saved_pie = {
    "Simulation": [],
    "Reality": [],
    "Ground-truth": []
}

function make_pie(svg, data, x, y, label) {

    let temp = {}
    data.map((d, i) => {
        temp[i] = 360 / n_portions;
        return d
    });


    const g = svg.append('g')
        .attr('transform', 'translate(' + x + ',' + y + ')');

    const pie = d3.pie()
        .value((d) => d[1]);


    const part = g.selectAll('.part')
        .data(pie(Object.entries(temp)))
        .enter()
        .append('g').append('path')
        .attr('d', d3.arc()
            .innerRadius(5)
            .outerRadius((_, i) => stats_orr_scale(data[i].length)))
        .attr("num", (_, i) => i)
        .attr("from", label)
        .attr('fill', () => {

            if (label === "Simulation") {
                return simCol
            }

            if (label === "Reality") {
                return realCol
            }
            return gtCol

        })

    g.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", stats_r)
        .attr("fill", "none")
        .attr("stroke", "#5555")
        .attr("stroke-width", "1px")

    g.append("text")
        .attr("x", 0)
        .attr("y", stats_r + 18)
        .style("text-anchor", "middle")
        .text(label)

}


function makeRealOrr() {

    let res = d3.range(n_portions).map(d => []);

    for (let i = 0; i < megaData[selMod]["real_dots"].length; i++) {
        let elem = megaData[selMod]["real_dots"][i];
        let theta = 360- elem["r"]
        let id = Math.round(theta / 360 * n_portions) % n_portions;
        res[id].push(elem["id"]) //+= 1
    }
    saved_pie["Reality"] = res;
    return res
}


function makeSimOrr() {

    let res = d3.range(n_portions).map(d => []);

    for (let i = 0; i < megaData[selMod]["sim_dots"].length; i++) {
        let elem = megaData[selMod]["sim_dots"][i];

        let theta = 360- elem["r"]

        let id = Math.round(theta / 360 * n_portions) % n_portions;
        res[id].push(elem["id"]) //+= 1
    }
    saved_pie["Simulation"] = res;
    return res
}


function makeGtOrr() {
    let res = d3.range(n_portions).map(d => []);

    for (let i = 0; i < megaData[selMod]["sim_dots"].length; i++) {
        let elem = megaData[selMod]["sim_dots"][i];

        let theta = 360- elem["gt_r"]
        let id = Math.round(theta / 360 * n_portions) % n_portions;
        res[id].push(elem["id"]) //+= 1
    }
    saved_pie["Ground-truth"] = res;
    return res
}


function makeBarData(data) {

    let res = d3.range(25).map(d => 0);

    for (let i = 0; i < data.length; i++) {

        res[Math.round(euclidian_dist([data[i]["x"], data[i]["y"]], [data[i]["gt_x"], data[i]["gt_y"]]))] += 1
    }

    return res
}


function makeBars() {

    const svg = d3.select("#stats_bar")

    svg.selectAll("*").remove();

    let sim = d3.range(17).map(d => [])
    let real = d3.range(17).map(d => [])
    let gt = d3.range(17).map(d => [])


    for (let i = 0; i < megaData[selMod]["sim_dots"].length; i++) {
        let elem = megaData[selMod]["sim_dots"][i];
        let elem2 = megaData[selMod]["real_dots"][i];

        let id = attrib(elem["x"], elem["y"]);
        let id2 = attrib(elem2["x"], elem2["y"]);
        let id3 = attrib(elem2["gt_x"], elem2["gt_y"]);

        if (id !== 15)
            sim[id].push(elem["id"]);
        if (id2 !== 15)
            real[id2].push(elem["id"]);
        if (id3 !== 15)
            gt[id3].push(elem["id"]);

    }


    id_stock["sim"] = sim;
    id_stock["real"] = real;
    id_stock["gt"] = gt;


    let res = sim.map((d, i) => (d.length > 0 ? i : -1)).filter(d => d > -1);
    res = res.concat(real.map((d, i) => (d.length > 0 ? i : -1)).filter(d => d > -1));
    res = res.concat(gt.map((d, i) => (d.length > 0 ? i : -1)).filter(d => d > -1));

    res = res.filter(onlyUnique);
    let stats_bar_scale = d3.scaleLinear().domain(d3.extent(sim.map(d => d.length))).range([0, 80]).clamp(true);

    let yland = 120
    let barW = 17
    let margW = 2

    if (res.length > 10) {
        barW = 180 / res.length;
    }

    let sim_anc = 120 - (barW + margW) * res.length / 2
    let gt_anc = 338 - (barW + margW) * res.length / 2
    let real_anc = 556 - (barW + margW) * res.length / 2


    svg.append("text")
        .attr("x", 340)
        .attr("y", 18)
        .style("text-anchor", "middle")
        .style("font-size", "14pt")
        .style("text-decoration", "underline")
        .text("Room Distribution")

    svg.selectAll('.simBar').append("g")
        .attr("class", "simBar statbar")
        .data(res)
        .enter()
        .append("rect")
        .attr("from", "sim")
        .attr("num", (d) => d)
        .attr("x", (_, i) => sim_anc + i * (barW + margW))
        .attr("y", (d) => yland - Math.max(stats_bar_scale(sim[d].length), 1))
        .attr("width", barW)
        .attr("height", d => Math.max(stats_bar_scale(sim[d].length), 1))
        .attr("fill", simCol)
        .attr("stroke", "#5555");


    svg.selectAll('.gtBar').append("g")
        .attr("class", "gtBar statbar")
        .data(res)
        .enter()
        .append("rect")
        .attr("from", "gt")
        .attr("num", (d) => d)
        .attr("x", (_, i) => gt_anc + i * (barW + margW))
        .attr("y", d => yland - Math.max(stats_bar_scale(gt[d].length), 1))
        .attr("width", barW)
        .attr("height", d => Math.max(stats_bar_scale(gt[d].length), 1))
        .attr("fill", gtCol)
        .attr("stroke", "#5555")


    svg.selectAll('.realBar').append("g")
        .attr("class", "realBar statbar")
        .data(res)
        .enter()
        .append("rect")
        .attr("from", "real")
        .attr("num", (d) => d)
        .attr("x", (_, i) => real_anc + i * (barW + margW))
        .attr("y", d => yland - Math.max(stats_bar_scale(real[d].length), 1))
        .attr("width", barW)
        .attr("height", d => Math.max(stats_bar_scale(real[d].length), 1))
        .attr("fill", realCol)
        .attr("stroke", "#5555")


    svg.selectAll('.realBar').append("g")
        .data(res)
        .enter()
        .append('text')
        .attr("text-anchor", "end")
        .style("transform", (_, i) => "translate(" + (sim_anc + i * (barW + margW) + (barW / 2) + 5) + "px," + (yland + 5) + "px) rotate(-85deg)")
        .style("font-weight", '400')
        .text((d) => roomLabels[d])


    svg.selectAll('.realBar').append("g")
        .data(res)
        .enter()
        .append('text')
        .attr("text-anchor", "end")
        .style("transform", (_, i) => "translate(" + (gt_anc + i * (barW + margW) + (barW / 2) + 5) + "px," + (yland + 5) + "px) rotate(-85deg)")
        .style("font-weight", '400')
        .text((d) => roomLabels[d])


    svg.selectAll('.realBar').append("g")
        .data(res)
        .enter()
        .append('text')
        .attr("text-anchor", "end")
        .style("transform", (_, i) => "translate(" + (real_anc + i * (barW + margW) + (barW / 2) + 5) + "px," + (yland + 5) + "px) rotate(-85deg)")
        .style("font-weight", '400')
        .text((d) => roomLabels[d])
}


function infront(x, y) {
    return contains([[6.4, -1.15], [14.15, 2.63]], [x, y])
}


function leftCor(x, y) {
    return contains([[12.5, 2.1], [14, 16.1]], [x, y])
}

function inBack(x, y) {
    return contains([[3.7, 16.01], [14.1, 18.04]], [x, y]) || contains([[3.97, 13.02], [7.3, 18.04]], [x, y])
}


function rightCor(x, y) {
    return contains([[5.7, 2.40], [7.3, 13.24]], [x, y])
}


function in12(x, y) {

    return contains([[1.9, 0.54], [5.8, 2.35]], [x, y])
}

function in2(x, y) {

    return contains([[-1.5, 2.56], [5.68, 5.75]], [x, y]) || contains([[-1.5, -0.9], [1.6, 3.49]], [x, y])
}

function in4(x, y) {

    return contains([[-1.5, 5.9], [5.74, 11.04]], [x, y])
}


function in5(x, y) {

    return contains([[-1.5, 11.2], [3.9, 16.4]], [x, y]) || contains([[2.3, 11.54], [5.75, 13]], [x, y])
}

function in6(x, y) {
    return contains([[14, 16], [18.92, 19.79]], [x, y])
}


function in7(x, y) {
    return contains([[14, 12.7], [19, 16.04]], [x, y])
}

function in8(x, y) {
    return contains([[14, 9.2], [19, 12.55]], [x, y])
}

function in9(x, y) {
    return contains([[14, 5.8], [19, 9.14]], [x, y])
}

function in10(x, y) {
    return contains([[14, 2.3], [19, 5.6]], [x, y])
}

function in11(x, y) {
    return contains([[14, -1.5], [19, 2.06]], [x, y])
}

function inMid(x, y) {
    return contains([[7.3, 2.6], [12.55, 15.9]], [x, y])

}

function attrib(x, y) {

    if (in12(x, y))
        return 12
    if (in2(x, y))
        return 2
    if (in4(x, y))
        return 4
    if (in5(x, y))
        return 5
    if (in6(x, y))
        return 6
    if (in7(x, y))
        return 7
    if (in8(x, y))
        return 8
    if (in9(x, y))
        return 9
    if (in10(x, y))
        return 10
    if (in11(x, y))
        return 11
    if (infront(x, y))
        return 1
    if (leftCor(x, y))
        return 3
    if (inBack(x, y))
        return 13
    if (rightCor(x, y))
        return 14
    if (inMid(x, y))
        return 0
    return 15


}

function onlyUnique(value, index, self) {
    return self.indexOf(value) === index;
}