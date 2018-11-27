var canvas, ctx, flag = false,
    prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    dot_flag = false;

var x = "black",
    y = 15;

var wating = false,
    label = null;

function init() {
    showMessage('Ingresa tu consulta', 'info');

    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}
init();

var points = [];

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = x;
    ctx.lineWidth = y;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.clearRect(0, 0, w, h);
}

function save() {
    showMessage('Realizando consulta...', 'info');
    document.getElementById("images-container").style.opacity = "0";
    sendQuery();
}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(currX, currY, 2, 2);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }

}

/*

Envia la query a la api de python con la imagen que saca del canvas

*/
function sendQuery() {
    canvas = document.getElementById('can');
    var base64 = canvas.toDataURL();
    var httpPost = new XMLHttpRequest(),
        path = "http://localhost:5000/query", // path de la api en python
        data = JSON.stringify({
            image: base64
        });
    httpPost.onreadystatechange = function (err) {
        if (httpPost.readyState == 4 && httpPost.status == 200) {
            json = JSON.parse(httpPost.responseText);
            images = json.images;
            showMessage("Es un ".concat(json.label), 'success');
            for (i = 0; i < 15; i++) {
                document.getElementById('res' + i).setAttribute(
                    'src', 'data:image/png;base64, '.concat(images[i])
                );
            }
            document.getElementById("images-container").style.opacity = "1";

            document.getElementById('query').setAttribute(
                'src', 'data:image/png;base64, '.concat(json.query)
            );
            // document.getElementById("images-container").style.display = "block";

        } else if(httpPost.readyState == 4){
            console.log(err);
            showMessage('Error en el servidor', 'error');
        }
    };
    // Set the content type of the request to json since that's what's being sent
    httpPost.open("POST", path, true);
    httpPost.setRequestHeader('Content-Type', 'application/json');
    httpPost.send(data);
};

/*
Metodo auxiliares para mostrar los mensajes
typ puede ser:

    *info (azul)
    *success (verde)
    *error (rojo)
*/

function showMessage(label, type) {
    let colorInfo = "#529cfc",
        colorSuccess = "#61d461",
        coloError = "#ff5e5e";

    document.getElementById('message').innerHTML = label;

    switch (type) {
        case 'success':
            document.getElementById('message').style.color = colorSuccess;
            break;
        case 'error':
            document.getElementById('message').style.color = coloError;
            break;
        case 'info':
            document.getElementById('message').style.color = colorInfo;
            break;
        default:
            document.getElementById('message').style.color = colorInfo;
            break;
    }
}