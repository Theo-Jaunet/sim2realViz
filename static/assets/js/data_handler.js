let combtype = 0;
let selInsts = [];

let selsteps = []

function merger(data) {

    if (keymap[18]) {
        let t_dat = d3.range(megaData[selMod]["real_dots"].length)
        data = $.grep(t_dat, function (value) {
            return $.inArray(value, data) < 0;
        });
    }

    if (keymap[16]) {
        merge_union(data)
    } else if (keymap[17]) {
        merge_inter(data)
    } else {
        merge_clear(data)
    }

    update_views(selInsts)
}

function merge_union(data) {

    // selsteps.push(data);
    selInsts = mergeDedupe([selInsts, data])
    // selInsts = [...new Set(selInsts, data)] //.concat(data).filter(onlyUnique)
    // console.log(selInsts);

    $("#nbInst").html(selInsts.length + "/" + megaData[selMod]["real_dots"].length)
}

function merge_inter(data) {
    selInsts = intersect(selInsts, data)
    // selInsts = [...new Set(selInsts, data)] //.concat(data).filter(onlyUnique)
    // console.log(selInsts);

    $("#nbInst").html(selInsts.length + "/" + megaData[selMod]["real_dots"].length)
}


function merge_clear(data) {
    selInsts = data

    $("#nbInst").html(selInsts.length + "/" + megaData[selMod]["real_dots"].length)
}


function intersect(a, b) {
    return a.filter(Set.prototype.has, new Set(b));
}

Array.prototype.unique = function () {
    let a = this.concat();
    for (let i = 0; i < a.length; ++i) {
        for (let j = i + 1; j < a.length; ++j) {
            if (a[i] === a[j])
                a.splice(j--, 1);
        }
    }
    return a;
};

const mergeDedupe = (arr) => {
    return [...new Set([].concat(...arr))];
}
