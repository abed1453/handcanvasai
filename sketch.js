let handPoseModel;
let webcamVideo;
let imageCanvas, webcamCanvas;
let imageCtx, webcamCtx;
let detections = [];
let lastIndexFingerPosition = { x: 0, y: 0 };
let lastThumbPosition = { x: 0, y: 0 };
let smoothedIndexX = 0;
let smoothedIndexY = 0;
let smoothedThumbX = 0;
let smoothedThumbY = 0;
const smoothingFactor = 0.3;

let viewImageMode = false;
let zoomInMode = false;
let findPersonsMode = false;

let originalImage = new Image();
const blurRadius = 8;
const rectWidth = 100;
const rectHeight = 100;

const minZoom = 1;
const maxZoom = 2;
let currentZoom = 1;
const zoomSensitivity = 0.002;

let cocoSsdModel;
let detectedPersons = [];

let modeActive = {
    viewImage: false,
    zoomIn: false,
    findPersons: false
};

function setup() {
    noCanvas();
    const imageContainer = select('#imageContainer');
    const imageElement = select('#staticImage');
    imageCanvas = select('#imageCanvas').elt;
    imageCtx = imageCanvas.getContext('2d');

    imageCanvas.width = 400;
    imageCanvas.height = 300;

    originalImage.src = imageElement.elt.src;
    originalImage.onload = () => {
        imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    };

    const webcamContainer = select('#webcamContainer');
    webcamVideo = select('#webcamVideo').elt;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcamVideo.srcObject = stream;
            webcamVideo.onloadedmetadata = () => {
                webcamVideo.play();
                initializeHandpose();
            };
        })
        .catch(err => {
            const errorMsg = createP('Error: Webcam access denied.');
            errorMsg.style('color', 'red');
            errorMsg.parent(webcamContainer);
        });

    webcamCanvas = select('#webcamCanvas').elt;
    webcamCtx = webcamCanvas.getContext('2d');

    webcamCanvas.width = 400;
    webcamCanvas.height = 300;

    window.addEventListener('keydown', (event) => {
        if (['v', 'V', 'z', 'Z', 'p', 'P', 'e', 'E'].includes(event.key)) {
            event.preventDefault();
        }
    }, false);

    window.addEventListener('keyup', (event) => {
        if (event.key === 'v' || event.key === 'V') {
            toggleViewImageMode();
        }
        if (event.key === 'z' || event.key === 'Z') {
            toggleZoomInMode();
        }
        if (event.key === 'p' || event.key === 'P') {
            toggleFindPersonsMode();
        }
        if (event.key === 'e' || event.key === 'E') {
            exitAllModes();
        }
    }, false);
}

function initializeHandpose() {
    handPoseModel = ml5.handpose(webcamVideo, () => {
        console.log('Handpose model loaded.');
    });

    handPoseModel.on('predict', results => {
        detections = results;
        if (detections.length > 0) {
            const indexFingerTip = detections[0].annotations.indexFinger[3];
            const thumbTip = detections[0].annotations.thumb[3];
            
            lastIndexFingerPosition.x = indexFingerTip[0];
            lastIndexFingerPosition.y = indexFingerTip[1];
            lastThumbPosition.x = thumbTip[0];
            lastThumbPosition.y = thumbTip[1];
        } else {
            lastIndexFingerPosition = { x: 0, y: 0 };
            lastThumbPosition = { x: 0, y: 0 };
        }
    });

    cocoSsdModel = ml5.objectDetector('cocossd', modelLoaded);
    
    function modelLoaded() {
        console.log('CocoSsd model loaded.');
    }
}

function toggleViewImageMode() {
    exitAllModes();
    modeActive.viewImage = true;
    viewImageMode = true;
    enterViewImageMode();
}

function toggleZoomInMode() {
    exitAllModes();
    modeActive.zoomIn = true;
    zoomInMode = true;
    enterZoomInMode();
}

function toggleFindPersonsMode() {
    exitAllModes();
    modeActive.findPersons = true;
    findPersonsMode = true;
    console.log("Starting person detection...");
    enterFindPersonsMode();
}

function enterViewImageMode() {
    select('#staticImage').addClass('blurred');
    clearCanvas(imageCtx);
    imageCtx.filter = `blur(${blurRadius}px)`;
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.filter = 'none';
}

function exitViewImageMode() {
    select('#staticImage').removeClass('blurred');
    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
}

function enterZoomInMode() {
    select('#staticImage').removeClass('blurred');
    currentZoom = 1;
    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
}

function exitZoomInMode() {
    currentZoom = 1;
    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
}

function enterFindPersonsMode() {
    select('#staticImage').removeClass('blurred');
    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    detectPersonsInImage();
}

function exitFindPersonsMode() {
    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    detectedPersons = [];
}

function exitAllModes() {
    if (modeActive.viewImage) {
        modeActive.viewImage = false;
        viewImageMode = false;
        exitViewImageMode();
    }
    if (modeActive.zoomIn) {
        modeActive.zoomIn = false;
        zoomInMode = false;
        exitZoomInMode();
    }
    if (modeActive.findPersons) {
        modeActive.findPersons = false;
        findPersonsMode = false;
        exitFindPersonsMode();
    }
}

function draw() {
    clearCanvas(webcamCtx);
    clearCanvas(imageCtx);  

    if (detections.length > 0) {
        smoothedIndexX = smoothingFactor * smoothedIndexX + (1 - smoothingFactor) * lastIndexFingerPosition.x;
        smoothedIndexY = smoothingFactor * smoothedIndexY + (1 - smoothingFactor) * lastIndexFingerPosition.y;
        smoothedThumbX = smoothingFactor * smoothedThumbX + (1 - smoothingFactor) * lastThumbPosition.x;
        smoothedThumbY = smoothingFactor * smoothedThumbY + (1 - smoothingFactor) * lastThumbPosition.y;

        let flippedIndexX = webcamCanvas.width - smoothedIndexX;
        let flippedIndexY = smoothedIndexY;
        let flippedThumbX = webcamCanvas.width - smoothedThumbX;
        let flippedThumbY = smoothedThumbY;

        drawIndicator(webcamCtx, flippedIndexX, flippedIndexY, 'red');

        if (modeActive.zoomIn) {
            drawIndicator(webcamCtx, flippedThumbX, flippedThumbY, 'blue');
            drawZoomedImage(flippedIndexX, flippedIndexY, flippedThumbX, flippedThumbY);
        } else if (modeActive.viewImage) {
            drawUnblurredRectangle(flippedIndexX, flippedIndexY);
        } else if (modeActive.findPersons) {
            findAndEnlargePerson(flippedIndexX, flippedIndexY);
        } else {
            imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
            drawIndicator(imageCtx, flippedIndexX, flippedIndexY, 'red');
        }
    } else {
        if (modeActive.viewImage) {
            enterViewImageMode();
        } else if (modeActive.zoomIn) {
            enterZoomInMode();
        } else if (modeActive.findPersons) {
            clearCanvas(imageCtx);
            imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
        } else {
            clearCanvas(imageCtx);
            imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
        }
    }
}

function clearCanvas(ctx) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function drawIndicator(ctx, x, y, color = 'red') {
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, 2 * Math.PI);
    ctx.fillStyle = color === 'red' ? 'rgba(255, 0, 0, 0.7)' : 'rgba(0, 0, 255, 0.7)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'white';
    ctx.stroke();
}

function drawUnblurredRectangle(x, y) {
    let rectX = x - rectWidth / 2;
    let rectY = y - rectHeight / 2;

    rectX = Math.max(0, Math.min(rectX, imageCanvas.width - rectWidth));
    rectY = Math.max(0, Math.min(rectY, imageCanvas.height - rectHeight));

    clearCanvas(imageCtx);
    imageCtx.filter = `blur(${blurRadius}px)`;
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.filter = 'none';

    imageCtx.save();
    imageCtx.beginPath();
    imageCtx.rect(rectX, rectY, rectWidth, rectHeight);
    imageCtx.clip();
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.restore();

    imageCtx.strokeStyle = 'white';
    imageCtx.lineWidth = 2;
    imageCtx.strokeRect(rectX, rectY, rectWidth, rectHeight);
}

function drawZoomedImage(indexX, indexY, thumbX, thumbY) {
    const midX = (indexX + thumbX) / 2;
    const midY = (indexY + thumbY) / 2;
    const distance = Math.hypot(thumbX - indexX, thumbY - indexY);
    currentZoom = 1 + (distance * zoomSensitivity);
    currentZoom = Math.min(Math.max(currentZoom, minZoom), maxZoom);

    const zoomWidth = imageCanvas.width / currentZoom;
    const zoomHeight = imageCanvas.height / currentZoom;
    let zoomX = midX - zoomWidth / 2;
    let zoomY = midY - zoomHeight / 2;

    zoomX = Math.max(0, Math.min(zoomX, imageCanvas.width - zoomWidth));
    zoomY = Math.max(0, Math.min(zoomY, imageCanvas.height - zoomHeight));

    clearCanvas(imageCtx);
    
    imageCtx.globalAlpha = 0.3;
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.globalAlpha = 1.0;

    imageCtx.drawImage(
        originalImage,
        zoomX, zoomY, zoomWidth, zoomHeight,
        0, 0, imageCanvas.width, imageCanvas.height
    );

    const sourceRect = {
        x: (zoomX / imageCanvas.width) * imageCanvas.width,
        y: (zoomY / imageCanvas.height) * imageCanvas.height,
        width: (zoomWidth / imageCanvas.width) * imageCanvas.width,
        height: (zoomHeight / imageCanvas.height) * imageCanvas.height
    };

    imageCtx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    imageCtx.lineWidth = 2;
    imageCtx.strokeRect(sourceRect.x, sourceRect.y, sourceRect.width, sourceRect.height);

    const adjustedIndexX = (indexX - zoomX) * currentZoom;
    const adjustedIndexY = (indexY - zoomY) * currentZoom;
    const adjustedThumbX = (thumbX - zoomX) * currentZoom;
    const adjustedThumbY = (thumbY - zoomY) * currentZoom;

    drawIndicator(imageCtx, adjustedIndexX, adjustedIndexY, 'red');
    drawIndicator(imageCtx, adjustedThumbX, adjustedThumbY, 'blue');
}

function detectPersonsInImage() {
    if (cocoSsdModel && originalImage.complete) {
        console.log("Starting person detection...");
        cocoSsdModel.detect(originalImage, (err, results) => {
            if (err) {
                console.error('Detection error:', err);
                return;
            }
            detectedPersons = results.filter(obj => obj.label === 'person');
            console.log(`Found ${detectedPersons.length} persons in the image:`, detectedPersons);
            
            clearCanvas(imageCtx);
            imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
            
            detectedPersons.forEach(person => {
                const scaledX = (person.x * imageCanvas.width) / originalImage.width;
                const scaledY = (person.y * imageCanvas.height) / originalImage.height;
                const scaledWidth = (person.width * imageCanvas.width) / originalImage.width;
                const scaledHeight = (person.height * imageCanvas.height) / originalImage.height;
                
                imageCtx.strokeStyle = 'red';
                imageCtx.lineWidth = 2;
                imageCtx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
            });
        });
    }
}

function findAndEnlargePerson(x, y) {
    if (detectedPersons.length === 0) {
        clearCanvas(imageCtx);
        imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);
        return;
    }

    clearCanvas(imageCtx);
    imageCtx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height);

    detectedPersons.forEach(person => {
        const scaledX = (person.x * imageCanvas.width) / originalImage.width;
        const scaledY = (person.y * imageCanvas.height) / originalImage.height;
        const scaledWidth = (person.width * imageCanvas.width) / originalImage.width;
        const scaledHeight = (person.height * imageCanvas.height) / originalImage.height;
        
        const expandedArea = 20;
        
        if (x >= (scaledX - expandedArea) && 
            x <= (scaledX + scaledWidth + expandedArea) &&
            y >= (scaledY - expandedArea) && 
            y <= (scaledY + scaledHeight + expandedArea)) {

            imageCtx.save();
            imageCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            imageCtx.fillRect(scaledX + 5, scaledY + 5, scaledWidth, scaledHeight);
            imageCtx.restore();

            imageCtx.save();
            
            imageCtx.beginPath();
            imageCtx.rect(scaledX - 2, scaledY - 2, scaledWidth + 4, scaledHeight + 4);
            imageCtx.clip();

            const scaleOffset = 0.08;
            const scaledPersonWidth = scaledWidth * (1 + scaleOffset);
            const scaledPersonHeight = scaledHeight * (1 + scaleOffset);
            const offsetX = (scaledPersonWidth - scaledWidth) / 2;
            const offsetY = (scaledPersonHeight - scaledHeight) / 2;

            imageCtx.drawImage(
                originalImage,
                person.x, person.y, person.width, person.height,
                scaledX - offsetX, scaledY - offsetY,
                scaledPersonWidth, scaledPersonHeight
            );

            imageCtx.restore();

            imageCtx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            imageCtx.lineWidth = 3;
            imageCtx.strokeRect(scaledX - 2, scaledY - 2, scaledWidth + 4, scaledHeight + 4);

            imageCtx.strokeStyle = 'red';
            imageCtx.lineWidth = 3;
            imageCtx.strokeRect(scaledX - 2, scaledY - 2, scaledWidth + 4, scaledHeight + 4);

        } else {
            imageCtx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
            imageCtx.lineWidth = 2;
            imageCtx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        }
    });
}
