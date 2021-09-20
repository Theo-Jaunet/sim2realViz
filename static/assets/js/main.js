const colScale = d3.scaleOrdinal([], d3.schemePaired)
let simW = [-1.6, 18.93];
let simH = [-1.3, 19.8];

// let simW = [1.1, -19.8];
// let simH = [-1.4, 18.4];

const mapSaleX = d3.scaleLinear().domain(simW).range([0, 650]);
const mapScaleY = d3.scaleLinear().domain(simH).range([0, 650]);
let megaData = load_data_light().then(d => save_data(d));
let selMod = 0
let dat_vals = [false, true, false]

let feat_bool = true;
let cam_bool = false
let occlu_bool = false;

let clip_bool = true;

let kmods = 2;

let real_curve = false;
let real_straight = false;

let sim_curve = false;
let sim_straight = false;


let simCol = "#99a8f5";
let realCol = "#f7a39e";
let gtCol = "#e6b564";

let modelCol = "#666666";

let glyph_simCol = "#787ef5";
let glyph_realCol = "#e07864";
let glyph_overlap = "#82e286";

let traj_mod = 0;
let adj_bool_real = false;
let adj_bool_sim = false;

let tot_dat;

let scaleMark = 2;

let added = [];

const og_nb = 4;

let mod_names = ["Vanilla", "DataAug", "Fine-tuned", "Perlin"];

let alphaPara = 0.15;

let heat_thresh = 0.05

function save_data(data) {
    tot_dat = data
    megaData = data[traj_mod]
    draw_models(d3.select("#overview"), megaData)
    draw_dots(d3.select("#overview"), megaData, selMod)
    d3.selectAll('.modRect[num="' + selMod + '"]').attr("class", "modRect selRect")
    $("#nbInst").html(selInsts.length + "/" + megaData[selMod]["real_dots"].length)
}


function draw_models(svg, data) {
    let space = 20
    let width = parseInt(svg.style("width").replace("px", ""))

    let vspace = 20
    let y = 30

    let xScale = d3.scaleLinear([0, 1], [0, width / 2 - space])
    // let xScale = d3.scaleLinear([1,0.2], [0, width / 2 - space])
    let xScale2 = d3.scaleLinear([1, 0], [width / 2 + space, width])
    // let xScale2 = d3.scaleLinear([0.2,1], [width / 2 + space, width])

    let rectSize = 12


// console.log(data);


    svg.append("line")
        .attr("id", "middleG")
        .attr("x1", width / 2)
        .attr("y1", 0)
        .attr("x2", width / 2)
        .attr("y2", 190)
        .attr("stroke-width", 1)
        .style("stroke", " #555555")
        .style("stroke-dasharray", ("3, 3"))

    svg.append("line")
        .attr("id", "middleG")
        .attr("x1", width / 2)
        .attr("y1", 230)
        .attr("x2", width / 2)
        .attr("y2", 400)
        .attr("stroke-width", 1)
        .style("stroke", " #555555")
        .style("stroke-dasharray", ("3, 3"))

    svg.selectAll(".modlines")
        .append("g")
        .attr("class", ".modlines")
        .data(data)
        .enter()
        .append("line")
        .attr("x1", (d, i) => 7.2 * mod_names[i].length)
        .attr("y1", (d, i) => vspace + (y * i) + y / 2 - 1)
        .attr("x2", width / 2 - space)
        .attr("y2", (d, i) => vspace + (y * i) + y / 2 - 1)
        .attr("stroke-width", 1)
        .style("stroke", " #555555")
        .style("opacity", " 0.6")


    svg.selectAll(".textLines")
        .append("g")
        .attr("class", "textLines")
        .data(data)
        .enter()
        .append("text")
        .attr("x", 5)
        .attr("y", (d, i) => vspace + (y * i) + y / 2 + 3)
        .text((_, i) => mod_names[i])
        .style("stroke", " #555555")


    svg.selectAll(".modlines")
        .append("g")
        .attr("class", ".modlines")
        .data(data)
        .enter()
        .append("line")
        .attr("x1", width / 2 + space)
        .attr("y1", (d, i) => vspace + (y * i) + y / 2 - 1)
        .attr("x2", (d, i) => width - (6.8 * mod_names[i].length))
        .attr("y2", (d, i) => vspace + (y * i) + y / 2 - 1)
        .attr("stroke-width", 1)
        .style("opacity", " 0.6")
        .style("stroke", " #555555");


    svg.selectAll(".textLines")
        .append("g")
        .attr("class", "textLines")
        .data(data)
        .enter()
        .append("text")
        .attr("x", width)
        .attr("y", (d, i) => vspace + (y * i) + y / 2 + 3)
        .text((_, i) => mod_names[i])
        .style("stroke", " #555555")
        .style("text-anchor", "end")


    svg.selectAll(".simDat")
        .append("g")
        .attr("class", "simDat")
        .data(data)
        .enter()
        .append("rect")
        .attr("num", (_, i) => i)
        .attr("width", rectSize)
        .attr("height", rectSize)
        .attr("class", "modRect")
        .attr("fill", modelCol)
        .style("transform", (d, i) => {
            return "translate(" + (xScale(d["sim_perf"]) + rectSize) + "px, " + (vspace + (y * i) + rectSize / 2) + "px) rotate(45deg) "
        })
        .style("stroke", " #555555")

    svg.selectAll(".realDat")
        .append("g")
        .attr("class", "realDat")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "modRect")
        .attr("num", (_, i) => i)
        .attr("width", rectSize)
        .attr("height", rectSize)
        .attr("fill", modelCol)
        .style("transform", (d, i) => {
            return "translate(" + (xScale2(d["real_perf"]) + rectSize) + "px, " + (vspace + (y * i) + rectSize / 2) + "px) rotate(45deg) "
        })
        .style("stroke", " #555555")


    let temp = added.filter(d => d ["traj"] === traj_mod);


    for (let i = 0; i < temp.length; i++) {


        svg.append("line")
            .attr("x1", width / 2 + space)
            .attr("y1", (d, i) => vspace + (y * (og_nb + i)) + y / 2 - 1)
            .attr("x2", width)
            .attr("y2", (d, i) => vspace + (y * (og_nb + i)) + y / 2 - 1)
            .attr("stroke-width", 1)
            .style("opacity", " 0.6")
            .style("stroke", " #555555");

        svg.append("line")
            .attr("x1", 10)
            .attr("y1", (d, i) => vspace + (y * (og_nb + i)) + y / 2 - 1)
            .attr("x2", width / 2 - space)
            .attr("y2", (d, i) => vspace + (y * (og_nb + i)) + y / 2 - 1)
            .attr("stroke-width", 1)
            .style("stroke", " #555555")
            .style("opacity", " 0.6")


        svg.append("rect")
            .attr("num", og_nb + i)
            .attr("width", rectSize)
            .attr("height", rectSize)
            .attr("class", "modRect")
            .attr("fill", modelCol)
            .style("transform",
                "translate(" + (xScale(temp[i]["data"]["sim_perf"]) + rectSize) + "px, " + (vspace + (y * (og_nb + i)) + rectSize / 2) + "px) rotate(45deg) "
            )
            .style("stroke", " #555555")

        svg.append("rect")
            .attr("class", "modRect")
            .attr("num", (og_nb + i))
            .attr("width", rectSize)
            .attr("height", rectSize)
            .attr("fill", modelCol)
            .style("transform", (d, i) => {
                return "translate(" + (xScale2(temp[i]["data"]["real_perf"]) + rectSize) + "px, " + (vspace + y * (og_nb + i) + rectSize / 2) + "px) rotate(45deg) "
            })
            .style("stroke", " #555555")


    }


// .style("stroke-dasharray", ("3, 3"))
//     makeArrow(svg, width / 2 - (space * 0.75), 120, 126)
//     makeArrow(svg, width / 2 + (space * 0.75), 120, 126)


    // svg.append("text")
    //     .attr("x", 0)
    //     .attr("y", 0)
    //     .style("text-anchor", "middle")
    //     .style("transform", "translate(" + (width / 2 + (space * 0.25)) + "px,195px) rotate(-90deg)")
    //     .text("Orrientation")


    // svg.append("text")
    //     .attr("x", 0)
    //     .attr("y", 14)
    //     .style("text-anchor", "middle")
    //     .style("transform", "translate(473px,200px) rotate(-90deg)")
    //     .text("Performance")


    // svg.append("text")
    //     .attr("x", (width / 2))
    //     .attr("y", 253)
    //     .style("text-anchor", "middle")
    //     .text("0")
    //
    //
    // svg.append("text")
    //     .attr("x", (width / 2))
    //     .attr("y", 145)
    //     .style("text-anchor", "middle")
    //     .text("100")

    // svg.append("line")
    //     .attr("x1", 0)
    //     .attr("x2", width / 2 - (space + 5))
    //     .attr("y1", 190 + vspace)
    //     .attr("y2", 190 + vspace)
    //     .attr("stroke", "#5555")
    //     .attr("stroke-width", "2px")

    svg.append("text")
        .attr("x", width / 2 - space / 2 + 10)
        .attr("y", 195 + vspace)
        .style("text-anchor", "middle")
        .style("font-size", "14pt")
        .style("text-decoration", "underline")
        .text("UMAP")

    // svg.append("line")
    //     .attr("x1", width)
    //     .attr("x2", width / 2 + (space + 5))
    //     .attr("y1", 190 + vspace)
    //     .attr("y2", 190 + vspace)
    //     .attr("stroke", "#5555")
    //     .attr("stroke-width", "2px")


    let g = svg.append("g")
        .attr("class", "arrow")

    let x = (width / 2 - space)
    y = 150 + vspace
    let length = 300

    g.append("line")
        .attr("x1", x + 1)
        .attr("x2", x - 7)
        .attr("y1", y)
        .attr("y2", y + 7)


    g.append("line")
        .attr("x1", x - 1)
        .attr("x2", x - 7)
        .attr("y1", y)
        .attr("y2", y - 7)


    g.append("line")
        .attr("x1", x)
        .attr("x2", x - length)
        .attr("y1", y)
        .attr("y2", y)


    svg.append("text")
        .attr("x", width * 0.25)
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("Accuracy")

    svg.append("text")
        .attr("x", width * 0.05)
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("0")

    svg.append("text")
        .attr("x", width * 0.45)
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("100");


    // H - REAL

    g = svg.append("g")
        .attr("class", "arrow")

    x = (width / 2 + 18);
    y = vspace + 150;
    length = 330;

    g.append("line")
        .attr("x1", x + 1)
        .attr("x2", x + 7)
        .attr("y1", y)
        .attr("y2", y + 7)


    g.append("line")
        .attr("x1", x - 1)
        .attr("x2", x + 7)
        .attr("y1", y)
        .attr("y2", y - 7);


    g.append("line")
        .attr("x1", x)
        .attr("x2", x + length)
        .attr("y1", y)
        .attr("y2", y)


    svg.append("text")
        .attr("x", (width * 0.75))
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("Accuracy")

    svg.append("text")
        .attr("x", width * 0.99)
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("0")

    svg.append("text")
        .attr("x", width * 0.55)
        .attr("y", y + 18)
        .style("text-anchor", "middle")
        .text("100")


    // Make Parallel

    y = 32

    let g_par = svg.append("g")

    for (let i = 0; i < data[0]["real_dots"].length; i++) {

        let temp = []
        // console.log(data[0]["real_dots"][i]);

        for (let j = 0; j < data.length; j++) {
            if (j < data.length - 1) {
                temp.push([xScale(data[j]["sim_dots"][i]["perf"]), (vspace + (y * j) + rectSize)])
            } else {
                temp.push([xScale(data[j]["sim_dots"][i]["perf"]), (vspace + (y * j) + (rectSize / 2))])

            }
        }


        let temp2 = temp.map(d => d[0])

        let base = temp2[0]
        temp2 = temp2.reduce((p, c) => p + c, 0) / temp2.length;

        let tscale = d3.scaleLinear().domain([0, 10]).range(['#75a945', 'red']);
        // if (Math.abs(base - temp2 > 15)) {

        // if (i > 20) break

        // console.log(temp);
        g_par.append("path")
            .data([{"diff": temp2, "line": temp, "id": data[0]["real_dots"][i]["id"]}])
            .attr("class", "sim_paraline")
            .attr("d", d => d3.line()(d["line"]))
            .attr("stroke", simCol)
            .attr("fill", "none")
            .data({"diff": temp2})
    }


    for (let i = 0; i < data[0]["real_dots"].length; i++) {

        let temp = []

        for (let j = 0; j < data.length; j++) {
            if (j < data.length - 1) {
                temp.push([xScale2(data[j]["real_dots"][i]["perf"]), (vspace + (y * j) + rectSize)])
            } else {
                temp.push([xScale2(data[j]["real_dots"][i]["perf"]), (vspace + (y * j) + (rectSize / 2))])
            }
        }

        let temp2 = temp.map(d => d[0]);

        let base = temp2[0];
        temp2 = temp2.reduce((p, c) => p + c, 0) / temp2.length;


        // if (Math.abs(base - temp2 > 15)) {

        // if (i > 20) break


        // console.log(temp);
        g_par.append("path")
            .data([{"diff": temp2, "line": temp, "id": data[0]["real_dots"][i]["id"]}])
            .attr("class", "real_paraline")
            .attr("d", d => d3.line()(d["line"]))
            .attr("stroke", realCol)
            .attr("fill", "none")


    }
    g_par.style("opacity", 0.6);
    g_par.moveToBack()

    let sat_svg = d3.select("#stats_orr");

    sat_svg.append("text")
        .attr("x", 340)
        .attr("y", 16)
        .style("text-anchor", "middle")
        .style("font-size", "14pt")
        .style("text-decoration", "underline")
        .text("Orientation")


}


function makeArrow(svg, x, y, length) {


    let g = svg.append("g")
        .attr("class", "arrow")


    g.append("line")
        .attr("x1", x + 1)
        .attr("x2", x - 7)
        .attr("y1", y)
        .attr("y2", y + 7)


    g.append("line")
        .attr("x1", x - 1)
        .attr("x2", x + 7)
        .attr("y1", y)
        .attr("y2", y + 7)


    g.append("line")
        .attr("x1", x)
        .attr("x2", x)
        .attr("y1", y)
        .attr("y2", y + length)

    // g.style("transform","rotate(90deg)")

}

function draw_dots(svg, data, mod) {


    let width = parseInt(svg.style("width").replace("px", ""))
    let tscale = d3.scaleLinear().domain([0, 10]).range(['#75a945', 'red']);

    let canvas = document.getElementById("ramp");
    ramp(canvas, tscale);
    canvas = document.getElementById("ramp_turbo");
    tscale = d3.scaleSequential(d3.interpolateSpectral).domain([10.05, 0]).clamp(true)
    ramp(canvas, tscale);
    tscale = d3.scaleLinear().domain([0, 10]).range(['#75a945', 'red']);
    let space = 20;

    let proj_simx = data[mod]["sim_dots"].map(d => d["proj"][0])
    let proj_simy = data[mod]["sim_dots"].map(d => d["proj"][1])


    let proj_realx = data[mod]["real_dots"].map(d => d["proj"][0])
    let proj_realy = data[mod]["real_dots"].map(d => d["proj"][1])

    // let xScale = d3.scaleLinear([0.2, 1], [0, width / 2 - space]).clamp(true).nice();
    let xScale2 = d3.scaleLinear(d3.extent(proj_realx), [width / 2 + space, width]).clamp(true).nice();
    // let xScale2 = d3.scaleLinear([1, 0.2], [width / 2 + space, width]).clamp(true).nice();
    let xScale = d3.scaleLinear(d3.extent(proj_simx), [0, width / 2 - space]).clamp(true).nice();

    //

    // let yScale = d3.scaleLinear([1, 0], [150, 240]).clamp(true).nice();
    let yScale2 = d3.scaleLinear(d3.extent(proj_realy), [230, 390]).clamp(true).nice();

    // let yScale2 = d3.scaleLinear([1, 0], [150, 240]).clamp(true).nice();
    let yScale = d3.scaleLinear(d3.extent(proj_simy), [230, 390]).clamp(true).nice();


    svg.selectAll("circle").remove()
    svg.selectAll("circle").remove()
    svg.selectAll(".realdots").remove()
    svg.selectAll(".simdots").remove()
    svg.selectAll(".brusher").remove()
    // svg.selectAll("g").remove()

    let xsim_tscale = d3.extent(proj_simx)
    let ysim_tscale = d3.extent(proj_simy)

    let xreal_tscale = d3.extent(proj_realx)
    let yreal_tscale = d3.extent(proj_realy)


    const brush = d3.brush()
        .extent([[xScale(xsim_tscale[0]), yScale(ysim_tscale[0])], [(xScale(xsim_tscale[1])), yScale(ysim_tscale[1])]])
        .on("start brush", ({selection}) => brushed2(selection));


    const brush2 = d3.brush()
        .extent([[xScale2(xreal_tscale[0]), yScale2(yreal_tscale[0])], [(xScale2(xreal_tscale[1])), yScale2(yreal_tscale[1])]])
        .on("start brush", ({selection}) => brushed(selection));

    svg.append("g")
        .attr("class", "brusher")
    // .call(brush); // todo: ICII

    svg.append("g")
        .attr("class", "brusher")
    // .call(brush2);


    let s_dots = svg.selectAll(".simdots")
        .append("g")
        .attr("class", "simdots")
        .data(data[mod]["sim_dots"])
        .enter()
        .append("circle")
        .attr("class", "simdots")
        .attr("num", d => d["id"])
        // .attr("cx", (d => xScale(d["perf"])))
        .attr("cx", (d => xScale(d["proj"][0])))
        .attr("cy", (d => yScale(d["proj"][1])))
        // .attr("cy", d => yScale(angleDist([d["gt_r"]], [d["r"]])))
        .attr("r", 4)
        .attr("fill", d => {

            if (kmods == 20) {
                return simCol
            } else {
                let treal = megaData[selMod]["real_dots"][d["id"]]
                let tsim = megaData[selMod]["sim_dots"][d["id"]]

                let tdist = euclidian_dist([treal["x"], treal["y"]], [tsim["x"], tsim["y"]]);


                return tscale(tdist)
            }

        });


    let r_dots = svg.selectAll(".realdots")
        .append("g")
        .attr("class", "realdots")
        .data(data[mod]["real_dots"])
        .enter()
        .append("circle")
        .attr("class", "realdots")
        .attr("num", d => d["id"])
        // .attr("cx", (d => xScale2(d["perf"])))
        // .attr("cy", d => yScale2(angleDist([d["gt_r"]], [d["r"]])))
        .attr("cx", (d => xScale2(d["proj"][0])))
        .attr("cy", (d => yScale2(d["proj"][1])))
        .attr("r", 4)
        .attr("fill", d => {

            if (kmods == 20) {
                return realCol
            } else {
                let treal = megaData[selMod]["real_dots"][d["id"]]
                let tsim = megaData[selMod]["sim_dots"][d["id"]]

                let tdist = euclidian_dist([treal["x"], treal["y"]], [tsim["x"], tsim["y"]]);


                return tscale(tdist)
            }

        });
    let sat_svg = d3.select("#stats_orr");

    sat_svg.selectAll("*").remove();
    sat_svg.append("text")
        .attr("x", 340)
        .attr("y", 16)
        .style("text-anchor", "middle")
        .style("font-size", "14pt")
        .style("text-decoration", "underline")
        .text("Orientation")


    let sim = makeSimOrr();
    make_pie(sat_svg, sim, 120, 90, "Simulation")

    let gt = makeGtOrr();
    make_pie(sat_svg, gt, 338, 90, "Ground-truth")

    let rea = makeRealOrr();
    make_pie(sat_svg, rea, 556, 90, "Reality")

    makeBars()


    // console.log(data[mod]["real_dots"]);
    if (kmods != 1) {
        fillMap(d3.select("#main"), data[mod]["real_dots"]);
    } else {
        fillMapGlyph(d3.select("#main"), data[mod]["real_dots"]);
    }

    function brushed(selection) {

        let ids = []
        d3.selectAll(".mapDot").remove()
        let save = d3.selectAll(".selSim");

        r_dots.attr("class", selection && (d => {

            let bool = contains(selection, [xScale2(d["proj"][0]), yScale2(d["proj"][1])])
            // let bool = contains(selection, [xScale2(d["perf"]), yScale(angleDist([d["gt_r"]], [d["r"]]))])


            return bool ? "selReal" : ""
        }));


        s_dots.attr("class", selection && ((d, i) => {
            let elem = megaData[selMod]["real_dots"][i]
            let bool = contains(selection, [xScale2(elem["proj"][0]), yScale2(elem["proj"][1])]);
            // let bool = contains(selection, [xScale2(elem["perf"]), yScale(angleDist([d["gt_r"]], [elem["r"]]))])
            // let bool = contains(selection, [xScale2(elem["perf"]), yScale(angleDist([d["gt_r"]], [elem["r"]]))])

            return bool ? "selReal" : ""
        }));


        save.attr("class", "selSim")


        // temp.attr("class","selReal")

        // d3.selectAll(".selReal").raise()

        const svg = d3.select("#main")
        let data = d3.selectAll('.selReal').data()
        let data2 = d3.selectAll('.selSim').data()


        if (kmods != 1) {
            fillMap(svg, data, "real");
        } else {
            fillMapGlyph(svg, data)
        }
        if (data2.length > 0)
            data = data.concat(data2)

        showPara(d3.select("#overview"), data)
    }


    function brushed2(selection) {

        let save = d3.selectAll(".selReal");

        // s_dots.attr("class", selection && (d => contains(selection, [xScale(d["perf"]), yScale(angleDist([d["gt_r"]], [d["r"]]))]) ? "selSim" : ""));
        s_dots.attr("class", selection && (d => contains(selection, [xScale(d["proj"][0]), yScale(d["proj"][1])]) ? "selSim" : ""));

        r_dots.attr("class", selection && ((d, i) => contains(selection, [xScale(megaData[selMod]["sim_dots"][i]["proj"][0]), yScale(megaData[selMod]["sim_dots"][i]["proj"][1])]) ? "selSim" : ""));
        // r_dots.attr("class", selection && ((d, i) => contains(selection, [xScale(megaData[selMod]["sim_dots"][i]["perf"]), yScale(angleDist([d["gt_r"]], [(megaData[selMod]["sim_dots"][i]["r"])]))]) ? "selSim" : ""));

        save.attr("class", "selReal")
        d3.selectAll(".mapDot").remove()
        const svg = d3.select("#main")
        let data2 = d3.selectAll('.selReal').data()
        let data = d3.selectAll('.selSim').data()

        let temp = data.map(d => d["id"])

        console.log(temp);
        if (kmods != 1) {
            fillMap(svg, data, "sim")
        } else {
            fillMapGlyph(svg, data)
        }

        if (data2.length > 0)
            data = data.concat(data2)
        showPara(d3.select("#overview"), data)
    }
}


async function drawFakeHeat(data) {

    // let data = d3.range(22).map(d => d3.range(22).map(z => Math.random()))

    // let data = await d3.json("static/data/simple.json")

    // console.log(data);

    let svg = d3.select("#main");

    d3.selectAll(".rect-tbrm").remove();

    let rep = 12;

    let col = d3.scaleSequential(d3.interpolateSpectral).domain([1.1, 0]).clamp(true)
    //
    // console.log(Math.max(...data.map(d => Math.max(...d))));
    if (occlu_bool) {
        console.log(data);
    }


    // let t_svg = d3.select("#clipper");
    // clipper.selectAll("*").remove()

    for (let i = 0; i < data.length; i++) {

        // console.log(Math.max(...data[i]));
        for (let j = 0; j < data[i].length; j++) {
            if (data[i][j] < 0) {
                console.log(data[i][j]);
            }


            if (data[i][j] > heat_thresh) {

                // clipper.append("circle")
                //     .attr("class", "rect-tbrm")
                //     .attr("cx", mapSaleX((i / rep) - 2) - 3)
                //     .attr("cy", mapScaleY(j / rep - 2))
                //     .attr("r",5)

                // .attr("width", mapSaleX((i / rep) + 1 / rep) - mapSaleX((i / rep)))
                // .attr("height", mapScaleY((i / rep) + 1 / rep) - mapScaleY((i / rep)))
                // .attr("fill", col(data[i][j]))


                // clipper.append("rect")
                //     .attr("class", "rect-tbrm")
                //     .attr("x", mapSaleX((i / rep) - 2) - 3)
                //     .attr("y", mapScaleY(j / rep - 2))
                //     .attr("width", mapSaleX((i / rep)  + 1 / rep) - mapSaleX((i / rep)))
                //     .attr("height", mapScaleY((i / rep) + 1 / rep) - mapScaleY((i / rep) ))
                //     // .attr("fill", col(data[i][j]))
                //     .attr("num", data[i][j])
                // .style("opacity", data[i][j])
                //
                svg.append("rect")
                    .attr("class", "rect-tbrm")
                    .attr("x", mapSaleX((i / rep) - 2) - 3)
                    .attr("y", mapScaleY(j / rep - 2))
                    .attr("width", mapSaleX((i / rep) - 2 + 1 / rep) - mapSaleX((i / rep) - 2))
                    .attr("height", mapScaleY((i / rep) - 2 + 1 / rep) - mapScaleY((i / rep) - 2))
                    .attr("fill", col(data[i][j]))
                    .attr("num", data[i][j])
                    .style("opacity", 0.9)
                .attr("fill",col(data[i][j]) )
                // .attr("stroke", "rgba(120,120,120,0.2)")
                // .attr("stroke-width", 1)
            }
        }
    }

}

function showPara(svg, data) {

    let width = parseInt(svg.style("width").replace("px", ""))

    let space = 20
    let vspace = 20

    let xScale = d3.scaleLinear([0.2, 1], [0, width / 2 - space]).clamp(true).nice();
    let xScale2 = d3.scaleLinear([1, 0.2], [width / 2 + space, width]).clamp(true).nice();

    let y = 30
    let rectSize = 12

    d3.selectAll(".paraSel").remove()
    let g_par = svg.append("g").attr("class", "paraSel")

    for (let i = 0; i < data.length; i++) {

        let temp = [];
        let temp2 = [];

        for (let j = 0; j < megaData.length; j++) {

            temp.push([xScale(megaData[j]["sim_dots"][data[i]["id"]]["perf"]), (vspace + (y * j) + rectSize)])
            temp2.push([xScale2(megaData[j]["real_dots"][data[i]["id"]]["perf"]), (vspace + (y * j) + rectSize)])
        }


        g_par.append("path")

            .attr("d", d3.line()(temp))
            .attr("stroke", "indigo")
            .style("stroke-width", "2px")
            .style("opacity", 0.8)
            .attr("fill", "none")


        g_par.append("path")
            .attr("d", d3.line()(temp2))
            .attr("stroke", "rgb(233, 150, 122)")
            .style("stroke-width", "2px")
            .style("opacity", 0.95)
            .attr("fill", "none")
    }

    /*
        for (let i = 0; i < data[0]["real_dots"].length; i++) {

            let temp = []

            for (let j = 0; j < data.length; j++) {
                temp.push([xScale2(data[j]["real_dots"][i]["perf"]), ((y * j) + rectSize)])
            }


            let temp2 = temp.map(d => d[0]);

            let base = temp2[0];
            temp2 = temp2.reduce((p, c) => p + c, 0) / temp2.length;


            if (Math.abs(base - temp2 > 15)) {

                // if (i > 20) break

                // console.log(temp);
                g_par.append("path")
                    .attr("d", d3.line()(temp))
                    .attr("stroke", "steelblue")
                    .attr("fill", "none")

            }


        }*/
}


function contains([[x0, y0], [x1, y1]], [x, y]) {
    return x >= x0 && x < x1 && y >= y0 && y < y1;
}

async function load_data_light() {

    return [[
        await d3.json('static/data/traj1/model1_proj.json', d3.autoType),
        await d3.json('static/data/traj1/model2_proj.json', d3.autoType),
        await d3.json('static/data/traj1/model3_proj.json', d3.autoType),
        // await d3.json('static/data/traj1/model5_proj.json', d3.autoType)
        // await d3.json('static/data/model4.json', d3.autoType),
        // await d3.json('static/data/traj1/model6_proj.json', d3.autoType)
        await d3.json('static/data/traj1/model_edited_proj.json', d3.autoType)
        // await d3.json('static/data/traj1/model6_proj_fine.json', d3.autoType)
        // await d3.json('static/data/traj1/model7_proj.json', d3.autoType)
        // await d3.json('static/data/traj1/model6_proj_conv.json', d3.autoType)
        // await d3.json('static/data/traj1/model6_proj_inv.json', d3.autoType)
        // await d3.json('static/data/traj1/model6_proj_abs.json', d3.autoType)
    ], [
        await d3.json('static/data/traj2/model1_proj.json', d3.autoType),
        await d3.json('static/data/traj2/model2_proj.json', d3.autoType),
        await d3.json('static/data/traj2/model3_proj.json', d3.autoType),
        await d3.json('static/data/traj2/model4_proj.json', d3.autoType)
    ]]
}


function euclidian_dist(a, b) {
    let sum = 0;

    for (let n = 0; n < a.length; n++) {
        sum += Math.pow(a[n] - b[n], 2)
    }
    return Math.sqrt(sum)
}

function angleDist(a, b) {

    let dist = (a - b + 360) % 360
    if (dist > 180)
        dist = 360 - dist
    return 1 - dist * (1 / 180)

    // return Math.pow(Math.cos(a * (1 / 360)) - Math.cos(b * (1 / 360)), 2) + Math.pow(Math.sin(a * (1 / 360)) - Math.sin(b * (1 / 360)), 2)
}


d3.selection.prototype.moveToFront = function () {
    return this.each(function () {
        this.parentNode.appendChild(this);
    });
};


function fillMapGlyph(svg, data) {
    // console.log("LALALA");
    svg.selectAll(".mapDottbrm").remove()
    svg.selectAll(".mapDot").remove()
    svg.selectAll(".glyphs").remove()


    let g = svg.append("g").attr("class", "glyphs")

    for (let i = 0; i < data.length; i++) {
        let sim = megaData[selMod]["sim_dots"][data[i]["id"]]
        let simx = mapSaleX(sim["x"])
        let simy = mapScaleY(sim["y"])


        let real = megaData[selMod]["real_dots"][data[i]["id"]]
        let realx = mapSaleX(real["x"])
        let realy = mapScaleY(real["y"])


        let gtx = mapSaleX(data[i]["gt_x"])
        let gty = mapScaleY(data[i]["gt_y"])

        drawInst(g, [[gtx, gty], [realx, realy], [simx, simy]], data[i]["id"])


        // console.log([[gtx, gty], [realx, realy], [simx, simy]]);


    }


}

function fillMap(svg, data, from) {

    let color = "lightseagreen";
    svg.selectAll(".glyphs").remove()
    let tscale = d3.scaleLinear().domain([0, 10]).range(['#75a945', 'red']);

    svg.selectAll(".mapDottbrm").remove()
    // svg.selectAll(".mapDot").remove()

    if (from === "sim") {
        color = "indigo"
    } else if (from === "real") {
        color = "darkred"
    }
    if (from === undefined) {
        svg.selectAll(".mapDot").remove()
    } else {
        svg.selectAll(".mapDot[from=" + from + "]").remove();
        svg.selectAll(".mapDot[fill='lightseagreen']").remove()
        svg.selectAll(".mapDot[fill='rgb(32, 178, 170)']").remove()
        svg.selectAll(".mapDot[fill='salmon']").remove()
    }


    for (let i = 0; i < data.length; i++) {


        if (dat_vals[0]) {


            let telem = megaData[selMod]["sim_dots"][data[i]["id"]]
            let tx = mapSaleX(telem["x"])
            let ty = mapScaleY(telem["y"])
            let tr = telem["r"]

            if (clip_bool)
                clipper.append("circle")
                    .attr("class", "mapDottbrm")
                    .attr("cx", tx)
                    .attr("cy", ty)
                    .attr("num", data[i]["id"])
                    .attr("r", clip_rad)


            // svg.append("circle")
            //     .attr("class", "mapDot")
            //     .attr("cx", tx)
            //     .attr("cy", ty)
            //     .attr("r", 5)
            //     .attr("fill", "rgb(162,138,210)")
            //     .style("stroke", "#555555")
            //     .attr("num", data[i]["id"])
            //     .attr("from", from)
            //     .attr("to", "sim_dots")

            svg.append("path")
                .attr("class", "mapDot")
                .attr("from", from)
                .attr("to", "sim_dots")
                .attr("num", data[i]["id"])
                .attr("stroke", "#555555")
                .attr("fill", simCol)
                .style("stroke-width", "1px")
                .attr("d", "m 20 20 a -6 6 7 0 1 14 0 l -7 23 l -7 -23")
                .attr("transform", "rotate(" + (180 - tr) + "  " + (tx - 2) + " " + (ty - 2) + ")  translate(" + (tx - 17) + " " + (ty - 17) + " )  scale(0.6)")

            if (dat_vals[1]) {

                let telem2 = data[i]
                let tx2 = mapSaleX(telem2["gt_x"])
                let ty2 = mapScaleY(telem2["gt_y"])

                svg.append("line")
                    .attr("class", "mapDottbrm")
                    .attr("x1", tx)
                    .attr("x2", tx2)
                    .attr("y1", ty)
                    .attr("y2", ty2)
                    .attr("num", data[i]["id"])
                    .style("stroke", "#555555")
                    .attr("stroke-width", "2px")
                    .style("opacity", 0.6)
            }


            if (dat_vals[2]) {

                let telem2 = megaData[selMod]["real_dots"][data[i]["id"]]
                let tx2 = mapSaleX(telem2["x"])
                let ty2 = mapScaleY(telem2["y"])


                svg.append("line")
                    .attr("class", "mapDottbrm")
                    .attr("x1", tx)
                    .attr("x2", tx2)
                    .attr("y1", ty)
                    .attr("y2", ty2)
                    .attr("num", data[i]["id"])
                    .style("stroke", "#555555")
                    .attr("stroke-width", "2px")
                    .style("opacity", 0.6)

            }


        }

        if (dat_vals[1]) {

            let telem = data[i]
            let tx = mapSaleX(telem["gt_x"])
            let ty = mapScaleY(telem["gt_y"])
            let tr = telem["gt_r"]


            //
            // svg.append("circle")
            //     .attr("class", "mapDot")
            //     .attr("cx", mapSaleX(data[i]["gt_x"]))
            //     .attr("cy", mapScaleY(data[i]["gt_y"]))
            //     .attr("r", 5)
            //     .style("stroke", "#555555")
            //     .attr("fill", colo$(this)r)
            //     .attr("num", data[i]["id"])
            //     .attr("from", from)
            //     .attr("to", "gt")

            if (clip_bool)
                clipper.append("circle")
                    .attr("class", "mapDottbrm")
                    .attr("cx", tx)
                    .attr("cy", ty)
                    .attr("r", clip_rad)

            let treal = megaData[selMod]["real_dots"][telem["id"]]
            let tsim = megaData[selMod]["sim_dots"][telem["id"]]

            let tdist = euclidian_dist([treal["x"], treal["y"]], [tsim["x"], tsim["y"]]);

            svg.append("path")
                .attr("class", "mapDot")
                .attr("num", data[i]["id"])
                .attr("from", from)
                .attr("to", "gt")
                .attr("stroke", "#555555")
                // .attr("fill", color)
                .attr("fill", () => (kmods == 0 ? gtCol : tscale(tdist)))
                .style("stroke-width", "1px")
                .attr("d", "m 20 20 a -6 6 7 0 1 14 0 l -7 23 l -7 -23")
                .attr("transform", "rotate(" + (180 - tr) + "  " + (tx - 2) + " " + (ty - 2) + ")  translate(" + (tx - 17) + " " + (ty - 17) + " )  scale(0.6)")

            if (dat_vals[2] && !dat_vals[0]) {

                let telem2 = megaData[selMod]["real_dots"][data[i]["id"]];
                let tx2 = mapSaleX(telem2["x"]);
                let ty2 = mapScaleY(telem2["y"]);


                svg.append("line")
                    .attr("class", "mapDottbrm")
                    .attr("x1", tx)
                    .attr("x2", tx2)
                    .attr("y1", ty)
                    .attr("y2", ty2)
                    .style("stroke", "#555555")
                    .attr("stroke-width", "2px")
                    .style("opacity", 0.6)

            }
        }

        if (dat_vals[2]) {


            let telem = megaData[selMod]["real_dots"][data[i]["id"]]
            let tx = mapSaleX(telem["x"])
            let ty = mapScaleY(telem["y"])
            let tr = telem["r"]


            if (clip_bool)
                clipper.append("circle")
                    .attr("class", "mapDottbrm")
                    .attr("cx", tx)
                    .attr("cy", ty)
                    .attr("r", clip_rad)

            //
            //
            // svg.append("circle")
            //     .attr("class", "mapDot")
            //     .attr("cx", mapSaleX(megaData[selMod]["real_dots"][data[i]["id"]]["x"]))
            //     .attr("cy", mapScaleY(megaData[selMod]["real_dots"][data[i]["id"]]["y"]))
            //     .attr("r", 5)
            //     .style("stroke", "#555555")
            //     .attr("fill", "salmon")
            //     .attr("num", data[i]["id"])
            //     .attr("from", from)
            //     .attr("to", "real_dots")
            //

            svg.append("path")
                .attr("class", "mapDot")
                .attr("num", data[i]["id"])
                .attr("from", from)
                .attr("to", "real_dots")
                .attr("stroke", "#555555")
                .attr("fill", "salmon")
                .style("stroke-width", "1px")
                .attr("d", "m 20 20 a -6 6 7 0 1 14 0 l -7 23 l -7 -23")
                .attr("transform", "rotate(" + (180 - tr) + "  " + (tx - 2) + " " + (ty - 2) + ")  translate(" + (tx - 17) + " " + (ty - 17) + " )  scale(0.6)")


            // if (dat_vals[1] && !dat_vals[0]) {
            //
            //     let telem2 = data[i]
            //     let tx2 = mapSaleX(telem2["gt_x"])-17
            //     let ty2 = mapScaleY(telem2["gt_y"])-17
            //
            //     svg.append("line")
            //         .attr("class", "mapDottbrm")
            //         .attr("x1", tx)
            //         .attr("x2", tx2)
            //         .attr("y1", ty)
            //         .attr("y2", ty2)
            //         .style("stroke", "#555555")
            //         .attr("stroke-width", "2px")
            //         .style("opacity", 0.6)
            //
            // }

        }
    }
    svg.selectAll("image").moveToBack()
}

d3.selection.prototype.moveToBack = function () {
    return this.each(function () {
        var firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};


// async function get_coords(coords, img) {
//
//     img = "data_source/out/sim/expe_man_ctrl/" + img
//     // coords = [20, 20];
//
//     console.log(img);
//
//     coords = coords.map(d => Math.round(d))
//
//     let form = new FormData();
//     form.append("img", img);
//     form.append("img_coord", coords);
//
//
//     $.ajax({
//         type: "POST",
//         url: "/coords",
//         processData: false,
//         contentType: false,
//         data: form,
//         success: function (d) {
//
//             console.log(d);
//
//             let temp = d3.select("#prev_depth")
//
//             temp.transition().duration(30)
//                 .attr("cx", mapSaleX(d[0]) - 15)
//                 .attr("cy", mapScaleY(d[2]))
//                 .style("opacity", 1)
//
//             temp.raise();
//             return [d[0], d[2]]
//         }
//     })
// }


function drawCam(data) {


    // console.log(data[]);

    $("#activ_sim_sal").attr("src", "data:image/jpeg;base64," + data["activ_sim_sal"])
    $("#activ_sim_depth_sal").attr("src", "data:image/jpeg;base64," + data["activ_sim_sal"])
    $("#activ_real_sal").attr("src", "data:image/jpeg;base64," + data["activ_real_sal"])
    $("#activ_real_depth_sal").attr("src", "data:image/jpeg;base64," + data["activ_real_sal"])

    $("#feat_sim_sal").attr("src", "data:image/jpeg;base64," + data["feat_sim_sal"])
    $("#feat_sim_depth_sal").attr("src", "data:image/jpeg;base64," + data["feat_sim_sal"])
    $("#feat_real_sal").attr("src", "data:image/jpeg;base64," + data["feat_real_sal"])
    $("#feat_real_depth_sal").attr("src", "data:image/jpeg;base64," + data["feat_real_sal"])


    $("#occlu_sim_sal").attr("src", "data:image/jpeg;base64," + data["occlu_sim_sal"])
    $("#occlu_sim_depth_sal").attr("src", "data:image/jpeg;base64," + data["occlu_sim_sal"])
    $("#occlu_real_sal").attr("src", "data:image/jpeg;base64," + data["occlu_real_sal"])
    $("#occlu_real_depth_sal").attr("src", "data:image/jpeg;base64," + data["occlu_real_sal"])

}


function drawOcclu(real_sal) {

    $("#real_occlu").attr("src", "data:image/jpeg;base64," + real_sal)
    $("#real_occlu_depth").attr("src", "data:image/jpeg;base64," + real_sal)

}

async function global_heat() {
    let form = new FormData()

    let meth = "dist";
    let prefix = "feat"
    let suffix = "add"
    if (cam_bool) {
        meth = "cam"

        prefix = "real"
        suffix = "avg"
    }
    //
    form.append("meth", meth);
    form.append("prefix", prefix);
    form.append("suffix", suffix);
    form.append("model", parseInt(selMod) + 1)
    form.append("traj", parseInt(traj_mod) + 1)

    // form.append("occlu", false);
    // form.append("cam", false);
    // form.append("feat", true);


    // $.ajax({
    //     type: "POST",
    //     url: "/global_mapping",
    //     processData: false,
    //     contentType: false,
    //     data: form,
    //     success: function (d) {
    //         drawFakeHeat(d)
    //     }
    // })


    $.ajax({
        type: "POST",
        url: "/fake_global",
        // url: "/global_mapping",
        processData: false,
        contentType: false,
        data: form,
        success: function (d) {
            drawFakeHeat(d)
        }
    })

}


function update_views(data) {

    // console.log(data);

    const svg = d3.select("#main")

    let data_real = megaData[selMod]["real_dots"].filter(d => data.includes(d["id"]))

    // let data_real = data.map(d => filterValue(, "id", d))
    let data2 = d3.selectAll('.selSim').data()

    // console.log(data_real);


    if (kmods != 1) {

        let marks = svg.selectAll(".mapDot")
        svg.selectAll(".mapDottbrm").style("visibility", "hidden")
        // marks.style("opacity", "0.05")
        marks.style("visibility", "hidden")


        for (let i = 0; i < data.length; i++) {
            // svg.selectAll(".mapDot[num='" + data[i] + "']").style("opacity", "1")
            svg.selectAll(".mapDot[num='" + data[i] + "']").style("visibility", "visible");

            svg.selectAll(".mapDottbrm[num='" + data[i] + "']").style("visibility", "visible").style("opacity", "0.6")
        }

        // fillMap(svg, data_real, "real");
    } else {

        let marks = svg.selectAll(".glyph")
        svg.selectAll(".mapDottbrm").style("visibility", "hidden")
        // marks.style("opacity", "0.05")
        marks.style("visibility", "hidden")
        // svg.selectAll(".slice").style("opacity", "0.05")


        for (let i = 0; i < data.length; i++) {
            svg.selectAll(".glyph[num='" + data[i] + "']").style("visibility", "visible");
            svg.selectAll(".mapDottbrm[num='" + data[i] + "']").style("opacity", "0.6")
        }


        // fillMapGlyph(svg, data_real)
    }

    //----------------- Parallel

    let sim_lines = d3.selectAll(".sim_paraline");

    sim_lines.style("opacity", '0.02')
    sim_lines.filter(d => data.includes(d["id"])).style("opacity", 0.8)


    let real_lines = d3.selectAll(".real_paraline")

    real_lines.style("opacity", '0.02')
    real_lines.filter(d => data.includes(d["id"])).style("opacity", 0.8)


    // --------------------- Proj

    d3.selectAll(".realdots").style("opacity", "0.6")
    d3.selectAll(".simdots").style("opacity", "0.6")


    for (let i = 0; i < data.length; i++) {
        d3.selectAll(".realdots[num='" + data[i] + "']").attr("class", "realdots selected").style("opacity", "1")
        d3.selectAll(".simdots[num='" + data[i] + "']").attr("class", "simdots selected").style("opacity", "1")

    }


}

function filterValue(obj, key, value) {
    return obj.find(function (v) {
        return v[key] === value
    });
}


function resetSel() {

    svg.selectAll(".mapDot").style("visibility", "visible")
    svg.selectAll(".glyph").style("visibility", "visible")
    svg.selectAll(".mapDottbrm").style("visibility", "visible").style("opacity", "0.6")
    d3.selectAll(".sim_paraline").style("opacity", "0.6")
    d3.selectAll(".real_paraline").style("opacity", "0.6")

}


function change_sim() {

    // let form = new FormData()


    let elem = megaData[selMod]["sim_dots"][selectedCR];
    let img = "x" + format(elem["gt_x"]) + "_y0.047" + "_z" + format(elem["gt_y"]) + "_r" + ((elem["gt_r"] > 180 ? elem["gt_r"] - 360 : elem["gt_r"])) + "_rgb.jpeg"

    let opts = {};
    opts["coords"] = {
        "x": format(elem["gt_x"]),
        "y": 0.047,
        "z": format(elem["gt_y"]),
        "o": ((elem["gt_r"] > 180 ? elem["gt_r"] - 360 : elem["gt_r"]))
    }


    opts["pitch"] = $("#s_pitch").val()
    opts["yaw"] = $("#s_yaw").val()
    opts["roll"] = $("#s_roll").val()
    opts["rgb_hfov"] = $("#s_rfov").val()
    opts["depth_hfov"] = $("#s_dfov").val()
    opts["mod"] = parseInt(selMod) + 1

    // form.append("params", );

    $.ajax({
        type: "POST",
        url: "/change_sim",
        processData: false,
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify(opts),
        success: function (d) {


            let rgb = d["rgb"]
            let depth = d["depth"]


            let svg = d3.select("#main")

            svg.selectAll(".prevTBRM").remove();


            let tr = d["res"]["r"];
            let tx = mapSaleX(d["res"]["x"]);
            let ty = mapScaleY(d["res"]["y"]);

            svg.append("path")
                .attr("class", "prevTBRM")
                .attr("stroke", "#555555")
                .attr("fill", "purple")
                .style("stroke-width", "1px")
                .attr("d", "m 20 20 a -6 6 7 0 1 14 0 l -7 23 l -7 -23")
                .attr("transform", "rotate(" + (180 - tr) + "  " + (tx - 2) + " " + (ty - 2) + ")  translate(" + (tx - 17) + " " + (ty - 17) + " )  scale(0.6)")


            $("#activ_sim_rgb").attr("src", "data:image/jpeg;base64," + rgb)
            $("#activ_sim_depth").attr("src", "data:image/jpeg;base64," + depth)


            $("#occlu_sim_rgb").attr("src", "data:image/jpeg;base64," + rgb)
            $("#occlu_sim_depth").attr("src", "data:image/jpeg;base64," + depth)


            $("#feat_sim_rgb").attr("src", "data:image/jpeg;base64," + rgb)
            $("#feat_sim_depth").attr("src", "data:image/jpeg;base64," + depth)

        }
    })


}