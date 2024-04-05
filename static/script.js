

let images = [];
let imagesName = [];
let currentImageIndex = 0;
let menuItemsCompleted = [];
let labels = [];
let prev_index = -1;
let change = false;
let detections = []; // 用來儲存所有圖片的檢測結果
let paddings = []
let ptype = 1;
let format_type = 'yolo'
const split_size = 480


document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('upload-button').addEventListener('click', function () {
        document.getElementById('file-input').click();
    });
    document.getElementById('file-input').addEventListener('change', handleFileSelect);
    document.getElementById('prev-button').addEventListener('click', showPrevImage);
    document.getElementById('next-button').addEventListener('click', showNextImage);
    document.getElementById('reset-button').addEventListener('click', labelreset);
    const ptypeBtns = document.querySelectorAll('.ptypeBtn');

    const liItems = document.querySelectorAll('li');

    liItems.forEach(li => {
        li.addEventListener('click', function() {
            ptype = parseInt(this.getAttribute('data-ptype'));  
            // 點到的變ptype active 其他的移除active
            liItems.forEach(item => {
                if (item === this) {
                    item.classList.add('ptype', 'active');
                } else {
                    item.classList.remove('ptype', 'active');
                }
            });
        });
    });

    document.getElementById('download-button').addEventListener('click', async function () {
        menuItemsCompleted.push(currentImageIndex)
        markImageAsCompleted(currentImageIndex); 
        downloadImage();
    });

    const formatSelect = document.getElementById('format-select');
    formatSelect.addEventListener('change', function() {
        format_type = formatSelect.value;
    });
});

function handleFileSelect(event) {
    const downloadStatus = document.getElementById('download-status')
    downloadStatus.textContent = '圖片上傳中'; 
    let files = Array.from(event.target.files);
    if (!files || files.length === 0) return;

    if (images.length >= 100) {
        alert("已達到圖片上限 無法繼續上傳，請先完成目前圖片標註！");
        return;
    }
    
    const promises = files.filter(f => f.type.match('image.*')).map(f => readFileAsDataURL(f));
    
    setTimeout(() => {
        downloadStatus.textContent = '已完成圖片上傳'; // 更新状态文本
    }, 10000);
    setTimeout(() => {
        downloadStatus.textContent = ''; // 清除状态文本
    }, 5000);
}

function readFileAsDataURL(file) {
    return new Promise((reject) => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                const [newImages, newImageNames] = splitImage(img, file, split_size);
                images.push(...newImages);
                imagesName.push(...newImageNames)
                updateImageMenu(imagesName); 
                showImage(currentImageIndex, false);
            };
            img.src = event.target.result;
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
    
}

function splitImage(image, file, size) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const numH = Math.ceil(image.height / size);
    const numW = Math.ceil(image.width / size);
    const newH = numH * size;
    const newW = numW * size;
    const padH = Math.ceil((newH - image.height) / 2);
    const padW = Math.ceil((newW - image.width) / 2);
    canvas.height = newH;
    canvas.width = newW;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, newW, newH);
    ctx.drawImage(
        image,
        0,
        0,
        image.width,
        image.height,
        padW,
        padH,
        image.width,
        image.height
    );
    const images = [];
    const imageNames = [];
    const fileName = file.name.split('.')[0];
    
    for (let h = 0; h < numH; h++) {
        for (let w = 0; w < numW; w++) {
            const cut_canvas = document.createElement('canvas');
            const cut_ctx = cut_canvas.getContext('2d');
            cut_canvas.width = size;
            cut_canvas.height = size;
            cut_ctx.drawImage(
                canvas,
                w * size,
                h * size,
                size,
                size,
                0,
                0,
                size,
                size,
            );
            const imageDataURL = cut_canvas.toDataURL();
            images.push(imageDataURL);
            const imageName = `${fileName}_h${h}_w${w}`;
            imageNames.push(imageName);
            detections.push([]); // Initialize detections array for each image
            const padding = createPadding(h, w, numH, numW, size, padH, padW)
            paddings.push([padding])
        }
    }
    return [images, imageNames];
}
function createPadding(h, w, numH, numW, size, padH, padW) {
    /*
    type (x, y -> px, py):
        0: >, >
        1: <, >
        2: >, <
        3: <, <
    */
    let padxmin = -1;
    let padymin = -1;
    let padxmax = -1;
    let padymax = -1;
    if (numH === 1 | numW === 1) {
       if (numH === 1) {
            if (numW === 1) { // num H = 1, numW = 1
                padxmin = padW
                padymin = padH
                padxmax = size-padW
                padymax = size-padH
            } else { // num H = 1, numW = n
                if (w === 0) {
                    padxmin = padW
                    padymin = padH
                    padxmax = size
                    padymax = size-padH
                } else if (w === numW-1) {
                    padxmin = 0
                    padymin = padH
                    padxmax = size-padW
                    padymax = size-padH
                } else {
                    padxmin = 0
                    padymin = padH
                    padxmax = size
                    padymax = size-padH
                    
                }
            }
       } else { // num H = n, numW = 1
            if (h === 0) {
                padxmin = padW
                padymin = padH
                padxmax = size-padW
                padymax = size
            } else if (h === numH-1) {
                padxmin = padW
                padymin = 0
                padxmax = size-padW
                padymax = size-padH
            } else {
                padxmin = padW
                padymin = 0
                padxmax = size-padW
                padymax = size
            }
       }
       
    } else if (h === 0) {
        if (w === numW-1) { // h = 0, w = numW-1
            padxmin = 0
            padymin = padH
            padxmax = size-padW
            padymax = size
        } else if (w === 0) { // h = 0, w = 0
            padxmin = padW
            padymin = padH
            padxmax = size
            padymax = size
        } else { // h = 0, w = k
            padxmin = 0
            padymin = padH
            padxmax = size
            padymax = size
        }
    } else if (h === numH-1) {
        if (w === numW-1) { // h = numH-1, w = numW-1
            padxmin = 0
            padymin = 0
            padxmax = size-padW
            padymax = size-padH
        } else if (w === 0) { // h = numH-1, w = 0
            padxmin = padW
            padymin = 0
            padxmax = size
            padymax = size-padH
        } else { // h = numH-1, w = k
            padxmin = 0
            padymin = 0
            padxmax = size
            padymax = size-padH
        }
    } else { 
        if (w === 0) { // h = k, w = 0
            padxmin = padW
            padymin = 0
            padxmax = size
            padymax = size
        } else if (w === numW - 1) { // h = k, w = numW-1
            padxmin = 0
            padymin = 0
            padxmax = size-padW
            padymax = size
        } else {
            padxmin = 0
            padymin = 0
            padxmax = size
            padymax = size
        }
    }
    return [padxmin, padymin, padxmax, padymax]
}
function notInPadding(paddings_list, x, y, bbox_size, ratio) {
    const xleft = (x) * ratio
    const ytop = (y) * ratio
    const xright =  (x) * ratio
    const ybottom = (y) * ratio

    const [padxmin, padymin, padxmax, padymax] = paddings_list[0]
    return xleft >= padxmin & ytop >= padymin & xright <= padxmax & ybottom <= padymax 
   
}
function showImage(index, change = true) {
    const container = document.getElementById('image-container');
    const imageDisplay = document.getElementById('image_display');
    if (imageDisplay) {
        container.removeChild(imageDisplay);
    }
    if (change & index !== prev_index) {
        container.innerHTML = '';
        labels = [];
        prev_index = currentImageIndex;
    }
    
    const img = document.createElement('img');
    img.id = 'image_display'
    // const img = document.getElementById('image-display');
    img.src = images[index];
    container.appendChild(img);
    detections.push([]); // Initialize detections array for each image
    // 更新菜单项样式
    const menuItems = document.querySelectorAll('#image-menu li');
    menuItems.forEach((menuItem, menuItemIndex) => {
        const link = menuItem.querySelector('a');
        if (menuItemIndex === index) {
            // 如果是当前显示的图像，则设置菜单项为蓝色字体并添加下划线
            link.style.color = 'blue';
            link.style.textDecoration = 'underline';
        } else {
            // 否则恢复默认样式
            link.style.color = 'black';
            link.style.textDecoration = 'underline';
        }
    });
    document.getElementById('image_display').addEventListener('click', imageClickHandler);
    updateImageCounter(currentImageIndex)
}


function showPrevImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        showImage(currentImageIndex);
    }
}

function showNextImage() {
    if (currentImageIndex < images.length - 1) {
        currentImageIndex++;
        showImage(currentImageIndex);
    }
}


function updateImageMenu(imageNames) {
    const menu = document.getElementById('image-menu');
    menu.innerHTML = ''; // 清空菜单内容

    // 为每张图像创建菜单项
    imageNames.forEach((name, index) => {
        const menuItem = document.createElement('li');
        menuItem.id = `menu-item-${index}`;

        // 添加超链接元素
        const link = document.createElement('a');
        link.textContent = name; // 使用图像名称作为超链接文本内容
        link.href = '#'; // 链接地址设为 #
        link.addEventListener('click', (event) => {
            event.preventDefault(); // 阻止默认点击事件
            if (images.length === 0) {
                showBlankImage();
            } else {
                currentImageIndex = index;
                showImage(index);
            }
        });
        menuItem.appendChild(link);

        const deleteButtonContainer = document.createElement('div'); // 创建一个新的容器元素
        deleteButtonContainer.style.display = 'inline-block'; // 设置容器为内联块级元素

        const deleteButton = document.createElement('button');
        deleteButton.textContent = '刪除圖片';
        deleteButton.className = 'delete-button';

        deleteButton.addEventListener('click', () => {
            deleteImage(index);
        });

        deleteButtonContainer.appendChild(deleteButton); // 将 delete button 放入容器内
        menuItem.appendChild(deleteButtonContainer); // 将容器放入菜单项内


        // 添加勾选框
        const checkbox = document.createElement('span');
        checkbox.className = 'checkbox';
        menuItem.appendChild(checkbox);
        // 检查图像是否已完成，如果是，则添加 completed 类
        if (menuItemsCompleted.includes(index)) {
            menuItem.classList.add('completed');
        }
        menu.appendChild(menuItem);
        
    });

    // 标记当前图像的菜单项
    const currentMenuItem = document.getElementById(`menu-item-${currentImageIndex}`);
    if (currentMenuItem) {
        // 将当前图像的菜单项设置为蓝色字体并添加下划线
        currentMenuItem.querySelector('a').style.color = 'blue';
        currentMenuItem.querySelector('a').style.textDecoration = 'underline';
    }
    
}

// 标记图像为已完成
function markImageAsCompleted(index) {
    const menuItem = document.getElementById(`menu-item-${index}`);
    if (menuItem) {
        menuItem.classList.add('completed'); // 添加已完成样式
    }
}

function deleteImage(index) {
    const menuItem = document.getElementById(`menu-item-${index}`);
    if (!menuItem) return;
    
    // 检查该图像是否已完成标记
    if (menuItem.classList.contains('completed')) {
        const completedIndex = menuItemsCompleted.indexOf(index);
        if (completedIndex !== -1) {
            menuItemsCompleted.splice(completedIndex, 1);
        }
        for (let i = 0; i < menuItemsCompleted.length; i++) {
            if (menuItemsCompleted[i] > index) {
                menuItemsCompleted[i] -= 1;
            }
        }
        // 直接删除图像和菜单项 
        images.splice(index, 1);
        imagesName.splice(index, 1);
        detections.splice(index, 1);
        paddings.splice(index, 1);
        updateImageMenu(imagesName);
        if (images.length === 0) {
            showBlankImage();
            return;
        } 
        if (currentImageIndex === index) {
            if (currentImageIndex !== 0) {
                currentImageIndex -= 1;
            }
        } else if (currentImageIndex >= index) {
            currentImageIndex -= 1;
        }
        
        showImage(currentImageIndex);
    } else {
        const confirmDelete = confirm('該圖像尚未完成標註 確定要進行刪除嗎？');
        for (let i = 0; i < menuItemsCompleted.length; i++) {
            if (menuItemsCompleted[i] > index) {
                menuItemsCompleted[i] -= 1;
            }
        }
        if (confirmDelete) {
            images.splice(index, 1);
            imagesName.splice(index, 1);
            detections.splice(index, 1);
            paddings.splice(index, 1);
            updateImageMenu(imagesName);
            if (images.length === 0) {
                showBlankImage();
                return;
            } 
            if (currentImageIndex === index) {
                if (currentImageIndex !== 0) {
                    currentImageIndex -= 1;
                }
            } else if (currentImageIndex >= index) {
                currentImageIndex -= 1;
            }

            showImage(currentImageIndex);
        }
    }
}

function showBlankImage() {
    const container = document.getElementById('image-container');
    container.innerHTML = '';
}

function imageClickHandler(event) {
    const classColors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'orange',
        5: 'pink',
        6: 'yellow',
        7: 'cyan',
    };

    const labelSizeInput = document.getElementById('label-size');
    const divSize = parseInt(labelSizeInput.value)

    change = true;
    const ratio = split_size / 720
    const rect = this.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    if (!notInPadding(paddings[currentImageIndex], clickX, clickY, divSize, ratio)) {
        return
    }
    const [xcenter, ycenter, width, height] = bboxAdjust(clickX, clickY, divSize, ratio, paddings[currentImageIndex])
    const divLeft = (xcenter - width / 2) ;
    const divTop = (ycenter - height / 2) ;
    const bboxWidth = width;
    const bboxHeight = height;
    
    const div = document.createElement('div');
    div.className = 'overlay-div';
    div.style.position = 'absolute';
    div.style.left = `${divLeft}px`;
    div.style.top = `${divTop}px`;
    div.style.width = `${bboxWidth}px`;
    div.style.height = `${bboxHeight}px`;
    div.style.backgroundColor = classColors[ptype - 1];
    div.style.opacity = '0.5';
    labels.push({
        id: ptype-1,
        x: xcenter * ratio / split_size,
        y: ycenter * ratio / split_size,
        w: bboxWidth * ratio / split_size,
        h: bboxHeight * ratio / split_size
    });

    /*
    const newDetection = [clickX - divSize, clickY - divSize, clickX + divSize, clickY + divSize, divSize, 0];
    detections[currentImageIndex].push(newDetection);
    */

    div.addEventListener('click', function () {
        this.remove();
        labels = labels.filter(label => {
            const labelwidth = label.w * rect.width;
            const labelheight = label.h * rect.height;
            const labelx = label.x * rect.width;
            const labely = label.y * rect.height;
            return !(Math.abs(clickX - labelx) * 2 < labelwidth && Math.abs(clickY - labely) * 2 < labelheight);
        });
        /*
        if (detections[currentImageIndex]) {
            detections[currentImageIndex] = detections[currentImageIndex].filter(detection => {
                // 检查点击的 x, y 是否在检测框范围内
                const [x0, y0, x1, y1] = detection;
                const withinXRange = clickX >= x0 && clickX <= x1;
                const withinYRange = clickY >= y0 && clickY <= y1;
                return !(withinXRange && withinYRange);
            });
        }
        */
    });

    document.getElementById('image-container').appendChild(div);
}

function updateImageCounter(index) {
    document.getElementById('image-counter').textContent = '圖片數量 ' + (index + 1) + ' / ' + images.length;
}

// reset all labels
function labelreset() {
    labels = []
    const imageContainer = document.getElementById('image-container');
    const divs = imageContainer.getElementsByClassName('overlay-div');
    while (divs.length > 0) {
        divs[0].parentNode.removeChild(divs[0]);
    }
}

function bboxAdjust(x, y, bbox_size, ratio, paddings_list) {
    let [padxmin, padymin, padxmax, padymax] = paddings_list[0]
    padxmin /= ratio
    padymin /= ratio
    padxmax /= ratio
    padymax /= ratio
    const xleft = (x - (bbox_size / 2)) 
    const ytop = (y - (bbox_size / 2)) 
    const xright =  (x + (bbox_size / 2)) 
    const ybottom = (y + (bbox_size / 2)) 
    let width = bbox_size
    let height = bbox_size 
    let xcenter = x
    let ycenter = y

    if (xleft < padxmin) {
        width = xright -padxmin
        xcenter = xright - width / 2
    } else if (xright > padxmax) {
        width = padxmax - xleft
        xcenter = xleft + width / 2
    }
    
    if (ytop < padymin) {
        height = ybottom - padymin
        ycenter = ybottom - height / 2
    } else if (ybottom > padymax) {
        height = padymax - ytop;
        ycenter = ytop + height / 2;
    }
    
    return [xcenter, ycenter, width, height]
}

async function downloadImage() {
    const downloadStatus = document.getElementById('download-status')
    if (images[currentImageIndex]) {
        const imageData = images[currentImageIndex]
        const imageName = imagesName[currentImageIndex]

        try {
            downloadStatus.textContent = `正在儲存 ${imageName} ...`;
            await fetch('/save_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image_data: imageData,
                    image_name: imageName,
                    format_type: format_type
                })
            });
            await fetch('/save_annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image_name: imageName, 
                    format_type: format_type,
                    yolo_labels: labels,
                    img_size:  split_size
                })
            });
            downloadStatus.textContent = '下載成功!';
            setTimeout(() => {
                downloadStatus.textContent = ''; // 清除下载状态文本
            }, 10000);
        } catch (error) {
            console.error('error: ', error)
            downloadStatus.textContent = '下載失敗.';
        }
    } else {
        downloadStatus.textContent = '沒有圖片可下載.';
    }
}

