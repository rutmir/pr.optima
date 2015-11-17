"use strict";

var source = null;
var currentIdx = 0;

var settings = {
    margin: {
        left: 10,
        right: 10,
        top: 3,
        bottom: 6
    },
    timeline: {
        font: "9px Calibri",
        line: {
            width: 3
        },
        height: 20
    },
    rate: {
        point: {
            radius: 10
        }
    },
    score: {
        font: "15px Calibri"
    },
    colorScheme1: {
        fg: "#000",
        bg: "#fff",
        successBg: "#efe",
        errorBg: "#fee",
        waitBg: "#ffe",
        pnt: "#f9f",
        flg: "#f66"
    }
};

function calculateDistance(rateArr) {
    var max = Number.MIN_VALUE;
    var min = Number.MAX_VALUE;

    rateArr.forEach(function (item) {
        if (item > max) {
            max = item;
        }
        if (item < min) {
            min = item;
        }
    });

    return {max: max, min: min, valid: max >= min, scale: 1};
}

function drawPoint(context, x, y, radius, cs) {
    context.save();

    context.lineWidth = radius / 3;

    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI, false);
    context.fillStyle = cs.bg;
    context.fill();
    context.strokeStyle = cs.pnt;
    context.stroke();
    context.beginPath();
    context.arc(x, y, radius / 3, 0, 2 * Math.PI, false);
    context.fillStyle = cs.fg;
    context.fill();

    context.restore();
}

function drawFlag(context, x, y, width, height, cs) {
    context.save();

    context.setLineDash([]);
    context.lineWidth = 1;
    context.strokeStyle = cs.fg;
    context.fillStyle = cs.flg;

    context.beginPath();
    context.arc(x, y, width - (width / 4), 0, Math.PI, false);
    context.closePath();
    context.fill();
    context.stroke();

    context.beginPath();
    context.moveTo(x - 0.5, y - 0.5);
    context.lineTo(x - 0.5, y - height - 0.5);
    context.lineTo(x - 0.5 + width, y - (3 * height / 4) - 0.5);
    context.lineTo(x - 0.5, y - (height / 2) - 0.5);
    context.closePath();
    context.fill();
    context.stroke();

    context.restore();
}

function drawScore(context, x, y, width, height, font, cs, base, precent) {
    var lineWidth1 = 5;
    context.save();

    context.lineWidth = lineWidth1;
    context.strokeStyle = cs.fg;
    context.fillStyle = cs.fg;

    var y0 = y + height - (width / 3);
    context.beginPath();
    context.moveTo(x + 0.5, y0 - 0.5);
    context.lineTo(x + width - 0.5, y0 - 0.5);
    context.stroke();

    context.lineWidth = 1;
    context.beginPath();
    context.arc(x + (width / 2), y0 - 0.5, (width / 3), 0, Math.PI, false);
    context.closePath();
    context.fill();
    context.stroke();

    context.beginPath();
    context.moveTo(x + (width / 10) - 0.5, y0 - 0.5);
    context.lineTo(x + (width / 10) - 0.5, ((2 * width) / 5));
    context.arc(x + (width / 2) + 0.5, ((2 * width) / 5) + 1.5, ((2 * width) / 5) + 1, Math.PI, 0, false);
    context.lineTo(x + ((9 * width ) / 10) + 0.5, y0 - 0.5);
    context.stroke();

    context.font = font;
    context.fillStyle = cs.bg;
    var baseW = context.measureText(base).width;
    context.fillText(base, x + (width / 2) - (baseW / 2), y0 + lineWidth1 + 7 - 0.5);

    var perc = Math.round(precent * 100);

    if (perc < 55) {
        context.fillStyle = cs.errorBg;
    } else if (perc < 65) {
        context.fillStyle = cs.waitBg;
    } else {
        context.fillStyle = cs.successBg;
    }
    var top = y0 * (1 - precent);
    var hR = y0 - top - lineWidth1 / 2;
    context.fillRect(x + (width / 10) + 0.5, top - 0.5, ((8 * width ) / 10) - 0.5, hR);

    context.fillStyle = cs.fg;
    var tx = perc + "%";
    var percW = context.measureText(tx).width;
    context.fillText(tx, x + (width / 2) - (percW / 2), top + hR / 2 + 3);

    context.restore();
}

function drawTimeline(context, width, height, offsetY, marginLeft, marginRight, cs, font, obj) {
    var time = [];
    obj.time.forEach(function (item) {
        time.push(new Date(item * 1000));
    });

    var needExtend = obj.result < 0;
    var len = time.length;
    var stepSize = (width - marginLeft - marginRight) / (len + (needExtend ? 1 : 0));
    var datesRange = {};
    context.save();
    context.font = font;

    for (var i = 0; i < len; i++) {
        var dt = time[i].getDate();
        datesRange[dt] = datesRange[dt] || [];
        datesRange[dt].push(i);

        var tm = time[i].getHours() + ":" + time[i].getMinutes();
        var txtM = context.measureText(tm);
        var offsetX = marginLeft + stepSize * i;

        if (!needExtend && i === len - 1 && obj.prediction >= 0) {
            var half = (obj.levels - 1 ) / 2;
            if (obj.result === obj.prediction || (obj.result < half && obj.prediction < half) || (obj.result > half && obj.prediction > half)) {
                context.fillStyle = cs.successBg;
            } else {
                context.fillStyle = cs.errorBg;
            }
            context.fillRect(offsetX, 0, offsetX + stepSize, height);
        }
        context.fillStyle = cs.fg;
        context.beginPath();
        context.moveTo(offsetX, offsetY + 0.5);
        context.lineWidth = 3;
        context.lineTo(offsetX + (stepSize / 2 - txtM.width / 2 - 2), offsetY + 0.5);
        context.stroke();

        context.fillText(tm, offsetX + (stepSize / 2 - txtM.width / 2), offsetY + 2.5);

        context.beginPath();
        context.moveTo(offsetX + (stepSize / 2 + txtM.width / 2 + 2), offsetY + 0.5);
        context.lineWidth = 3;
        context.lineTo(offsetX + stepSize, offsetY + 0.5);
        context.stroke();
    }
    if (needExtend) {
        var offsetX2 = marginLeft + stepSize * len;
        if (obj.prediction >= 0) {
            context.fillStyle = cs.waitBg;
            context.fillRect(offsetX2, 0, offsetX2 + stepSize, height);
        }
        context.fillStyle = cs.fg;
        context.beginPath();
        context.moveTo(offsetX2, offsetY + 0.5);
        context.lineWidth = 3;
        context.lineTo(offsetX2 + stepSize, offsetY + 0.5);
        context.stroke();
    }
    context.restore();
}

function drawRates(context, width, offsetY, marginTop, marginLeft, marginRight, cs, radius, obj) {
    var dist = calculateDistance(obj.data);
    if (dist.valid) {
        var i = 0;
        dist.scale = (offsetY - marginTop - radius ) / (dist.max - dist.min);
        var len = obj.data.length;
        var stepSize = (width - marginLeft - marginRight) / (len + (obj.result < 0 ? 1 : 0));
        var c = [];
        for (i = 0; i < len; i++) {
            c.push({
                x: marginLeft + stepSize / 2 + stepSize * i,
                y: offsetY - (obj.data[i] - dist.min) * dist.scale
            });
        }
        context.save();
        context.lineWidth = 1;
        context.strokeStyle = cs.fg;
        for (i = 0; i < len - 1; i++) {
            context.beginPath();
            context.moveTo(c[i].x, c[i].y);
            context.lineTo(c[i + 1].x, c[i + 1].y);
            context.stroke();
        }
        context.restore();
        for (i = 0; i < len; i++) {
            drawPoint(context, c[i].x, c[i].y, radius, cs);
        }
    }
}

function drawNeurons(context, width, offsetY, marginTop, marginLeft, marginRight, font, cs, radius, obj) {
    var verticalStep = (offsetY - marginTop - radius ) / (obj.levels - 1);
    var yArr = [];
    var i = 0;
    context.save();
    context.font = font;
    context.setLineDash([1, 3]);
    context.lineWidth = 1;
    context.strokeStyle = cs.fg;

    for (i = 0; i < obj.levels; i++) {
        var y = offsetY - verticalStep * i;
        yArr.push(y);
        var tx = "" + (i + 1);
        var txtW = context.measureText(tx).width;
        context.fillText(tx, marginLeft, y + 2.5);
        context.beginPath();
        context.moveTo(marginLeft + txtW + 3, y + 0.5);
        context.lineTo(width - marginRight, y + 0.5);
        context.stroke();
    }

    var len = obj.source.length;
    var stepSize = (width - marginLeft - marginRight) / (len + 1);
    var xArr = [];
    for (i = 0; i < len; i++) {
        xArr.push(marginLeft + stepSize / 2 + stepSize * i);
    }
    context.setLineDash([]);
    for (i = 0; i < len - 1; i++) {
        context.beginPath();
        context.moveTo(xArr[i], yArr[obj.source[i]]);
        context.lineTo(xArr[i + 1], yArr[obj.source[i + 1]]);
        context.stroke();
    }

    if (obj.prediction >= 0 && obj.result !== obj.prediction) {
        context.setLineDash([1, 3]);
        context.beginPath();
        context.moveTo(xArr[len - 1], yArr[obj.source[len - 1]]);
        context.lineTo(xArr[len - 1] + stepSize, yArr[obj.prediction]);
        context.stroke();
    }

    if (obj.result >= 0) {
        context.setLineDash([]);
        context.beginPath();
        context.moveTo(xArr[len - 1], yArr[obj.source[len - 1]]);
        context.lineTo(xArr[len - 1] + stepSize, yArr[obj.result]);
        context.stroke();
    }
    context.restore();
    for (i = 0; i < len; i++) {
        drawPoint(context, xArr[i], yArr[obj.source[i]], radius, cs);
    }
    if (obj.result >= 0) {
        drawPoint(context, xArr[len - 1] + stepSize, yArr[obj.result], radius, cs);
    }
    if (obj.prediction >= 0) {
        drawFlag(context, xArr[len - 1] + stepSize, yArr[obj.prediction], radius, radius * 2, cs);
    }
}

function printRatesGraph(canvas, obj, settings) {
    var x = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    x.clearRect(0, 0, w, h);

    drawTimeline(x, w, h, h - settings.margin.bottom, settings.margin.left, settings.margin.right, settings.colorScheme1, settings.timeline.font, obj);
    drawRates(x, w, h - settings.margin.bottom - 20, settings.margin.top, settings.margin.left, settings.margin.right, settings.colorScheme1, settings.rate.point.radius, obj);
}

function printNeuroGraph(canvas, obj, settings) {
    var x = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    x.clearRect(0, 0, w, h);

    drawTimeline(x, w, h, h - settings.margin.bottom, settings.margin.left, settings.margin.right, settings.colorScheme1, settings.timeline.font, obj);
    drawNeurons(x, w, h - settings.margin.bottom - 20, settings.margin.top, settings.margin.left, settings.margin.right, settings.timeline.font, settings.colorScheme1, settings.rate.point.radius, obj);
}

function printScoreGraf(canvas, s10, s100, settings) {
    var x = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    var ctrlWidth = w / 6;
    x.clearRect(0, 0, w, h);

    drawScore(x, 0, 0, ctrlWidth, h, settings.score.font, settings.colorScheme1, 10, s10);
    drawScore(x, w - ctrlWidth, 0, ctrlWidth, h, settings.score.font, settings.colorScheme1, 100, s100);
}

function refreshGrafs() {
    if (currentIdx < 0) {
        currentIdx = 0;
    }
    if (currentIdx >= source.data.length) {
        currentIdx = source.data.length - 1;
    }

    printRatesGraph(document.getElementById("drawRates"), source.data [currentIdx], settings);
    printNeuroGraph(document.getElementById("drawNeurons"), source.data [currentIdx], settings);
}

function refresh() {
    if (source !== null) {
        refreshGrafs();
        printScoreGraf(document.getElementById("drawScores"), source.score10, source.score100, settings);
    }
}

function loadSymbol(symbol) {
    $.get("https://rp-optima.appspot.com/api/json/" + symbol + "/all", function (data) {
        if (data !== null && data.data !== null) {
            source = data;
            currentIdx = data.data.length - 1;
            refresh();
        } else {
            window.alert("Error");
        }
    });
}

$(function () {
    $('#next').on('click', function () {
        if (source !== null) {
            currentIdx++;
            refreshGrafs();
        }
    });

    $('#prev').on('click', function () {
        if (source !== null) {
            currentIdx--;
            refreshGrafs();
        }
    });

    $('#currency').on('change', function () {
        loadSymbol($(this).val());
    });

    loadSymbol("eur");
});
