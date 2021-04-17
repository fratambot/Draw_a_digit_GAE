let canvas = document.getElementById('inputimage');
let ctx = canvas.getContext('2d');
let ctx2 = document.getElementById('predchart').getContext('2d');
let pred_digit = document.getElementById('pred_digit')
let pred_prob = document.getElementById('pred_prob')
let mouselbtn = false;

// initialize
window.onload = ()=>{
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  console.log('Hello geek :)')
};

// Mouse events
canvas.addEventListener("mousedown", (e) =>{
  if(e.button == 0){
    let rect = e.target.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    mouselbtn = true;
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
});

canvas.addEventListener("mousemove", (e) =>{
  let rect = e.target.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  if(mouselbtn){
      ctx.lineTo(x, y);
      ctx.stroke();
  }
});

canvas.addEventListener("mouseup", (e) =>{
  if(e.button == 0){
    mouselbtn = false;
    saveImage();
  }
});

// Tactile events
canvas.addEventListener("touchstart", (e)=>{
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(x, y);
    }
});

canvas.addEventListener("touchmove", (e)=>{
    if (e.targetTouches.length == 1) {
        let rect = e.target.getBoundingClientRect();
        let touch = e.targetTouches[0];
        let x = touch.clientX - rect.left;
        let y = touch.clientY - rect.top;
        ctx.lineTo(x, y);
        ctx.stroke();
        e.preventDefault();
    }
});

canvas.addEventListener("touchend", (e)=>saveImage());

// Clear all
document.getElementById("clearbtn").onclick = onClear;
function onClear(){
    mouselbtn = false;
    // clear canvas
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    // clar prediction and prob
    pred_digit.textContent = null;
    pred_prob.value = null;
    // clear chart
    predchart.data.datasets[0].data = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    // for some reason predchart.data.datasets[0].data.pop() is not working
    predchart.update();
};

// Ajax POST
function saveImage() {
  console.time("time");
  canvas.toBlob((blob) => {
    let form = new FormData();
    form.append('img', blob, "dummy.png"),

    $.ajax({
        url: "/prediction",
        type: "POST",
        data: form,
        processData: false,
        contentType: false,
    }).then(
        (data)=>showResults(JSON.parse(data)),
        ()=>alert("error")
    )
  })

  console.timeEnd("time");
};

function showResults(res) {
  pred_digit.textContent = res.pred;
  pred_prob.value = res.probs[res.pred].toFixed(2);
  data = res.probs;
  updateData(predchart, data);
};

function updateData(predchart, data) {
    predchart.data.datasets[0].data = data;
    predchart.update();
};

// Chart.js histogram
var predchart = new Chart(ctx2, {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: ['rgba(255, 0, 0, 0.2)'],
            borderColor: ['rgba(255, 0, 0, 1)'],
            borderWidth: 2,
            borderRadius: 3,
        }]
    },
    options: {
        scales: {
            x: {
              min: 0,
              max: 10,
              title: {
                text: "Class",
                display: true,
                color: 'rgba(255,0,0,1)'
              },
              grid: {
                  display: false
              },
            },
            y: {
                min: 0,
                max: 100,
                stepSize: 1,
                title: {
                  text: "Probability",
                  display: true,
                  color: 'rgba(255,0,0,1)'
                }
            }
        },
        maintainAspectRatio: false,
        plugins: {
            tooltip: {
                // Disable the on-canvas tooltip
                enabled: false
            },
            legend: {
                display: false
            }
        },
    }
});
