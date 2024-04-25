

let images = [];
let imagesName = [];
let currentImageIndex = 0;
let menuItemsCompleted = [];
let completedImageName = [];
let parentImagesName = [];
let labels = [];
let prev_index = -1;
let change = false;
let detections = []; 
let paddings = []
let ptype = 1;
let classSet = 1;
let annotations = [];
let format_type = 'yolo'
let mode ='annotate'
const split_size = 480
const classColors = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'purple',
    4: 'orange',
    5: 'pink',
    6: 'yellow',
    7: 'cyan',
    8: 'magenta',
    9: 'teal',
    10: 'brown',
    11: 'olive',
    12: 'navy'
};
let anno_ids = []

function submitForm(event) {
    event.preventDefault(); 

    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const login_set = document.getElementById('login-set');
    login_set.addEventListener('change', function() {
        classSet = login_set.value;
    });
    const login_mode = document.getElementById('login-mode');
    login_mode.addEventListener('change', function() {
        classSet = login_mode.value;
    });
    fetch('/website/validate_password', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: username,
            password: password
        })
    })
    .then(response => {
        if (response.ok) {
            
            document.getElementById("form_container").style.display = "none";
            document.querySelector(".content").style.display = "block";
            document.querySelector("nav").style.display = "block";
            localStorage.setItem('username', username);
            localStorage.setItem('login_set', classSet)
            localStorage.setItem('login_mode', mode)
            displaySet(classSet)
            displayMode(mode)
        } else {
            const loginStatus = document.getElementById('login-status')
            loginStatus.textContent = "Password Incorrect"
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
function displaySet(set) {
    if (classSet === 1) {
        document.getElementById("form_container").style.display = "none";
        document.querySelector(".content").style.display = "block";
        document.querySelector("nav").style.display = "block";
        classSet = 1;
        pagination1_1.style.display = 'block';
        pagination1_2.style.display = 'block';
        pagination2_1.style.display = 'none';
        pagination2_2.style.display = 'none';
        pagination2_3.style.display = 'none';
        
        const liItems = document.querySelectorAll('li');
        liItems.forEach(li => {
            li.classList.remove('ptype', 'active');
        });
        const class1ptype = document.getElementById('class1ptype');
        class1ptype.classList.add('ptype', 'active');
        ptype = 1;
        console.log("Set1")
    } else {
        document.getElementById("form_container").style.display = "none";
        document.querySelector(".content").style.display = "block";
        document.querySelector("nav").style.display = "block";
        classSet = 2;
        pagination1_1.style.display = 'none';
        pagination1_2.style.display = 'none';
        pagination2_1.style.display = 'block';
        pagination2_2.style.display = 'block';
        pagination2_3.style.display = 'block';

        const liItems = document.querySelectorAll('li');
        liItems.forEach(li => {
            li.classList.remove('ptype', 'active');
        });
        const class2ptype = document.getElementById('class2ptype');
        class2ptype.classList.add('ptype', 'active');
        ptype = 1;
    }
}
function displayMode(mode) {
    if (mode === 'annotate') {
        document.getElementById("upload-label-button").style.display = "none";
    } else if (mode === 'modify') {
        document.getElementById("upload-label-button").style.display = "block";
    } else {
        console.error()
    }
}
window.onload = function() {
    const username = localStorage.getItem('username');
    const login_set = localStorage.getItem('login_set');
    const login_mode = localStorage.getItem('login_mode');
    
    if (username) {
        document.getElementById("form_container").style.display = "none";
        document.querySelector(".content").style.display = "block";
        document.querySelector("nav").style.display = "block";
        classSet = login_set
        mode = login_mode
        displaySet(login_set)
        displayMode(login_mode)
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const login_set = document.getElementById('login-set');
    login_set.addEventListener('change', function() {
        classSet = login_set.value;
    });
    const login_mode = document.getElementById('login-mode');
    login_mode.addEventListener('change', function() {
        mode = login_mode.value;
    });
    document.getElementById("form_container").style.display = "block";
    document.getElementById('upload-button').addEventListener('click', function () {
        document.getElementById('file-input').click();
    });
    document.getElementById('file-input').addEventListener('change', handleFileSelect);
    document.getElementById('upload-label-button').addEventListener('click', function () {
        document.getElementById('label-input').click();
    });
    document.getElementById('label-input').addEventListener('change', handleLabelSelect);
    document.getElementById('prev-button').addEventListener('click', showPrevImage);
    document.getElementById('next-button').addEventListener('click', showNextImage);
    document.getElementById('reset-button').addEventListener('click', labelreset);
    const ptypeBtns = document.querySelectorAll('.ptypeBtn');

    const liItems = document.querySelectorAll('li');

    liItems.forEach(li => {
        li.addEventListener('click', function() {
            ptype = parseInt(this.getAttribute('data-ptype'));
            liItems.forEach(item => {
                if (item === this) {
                    item.classList.add('ptype', 'active');
                } else {
                    item.classList.remove('ptype', 'active');
                }
            });
        });
    });

    document.getElementById('complete-button').addEventListener('click', async function () {
        if (!completedImageName.includes(imagesName[currentImageIndex])) {
	    completedImageName.push(imagesName[currentImageIndex])
	}
        menuItemsCompleted.push(currentImageIndex)
        markImageAsCompleted(currentImageIndex); 
        complete_label_image();
    });

    document.getElementById('download-button').addEventListener('click', async function () {
        download_labeled_images()
    });

    const formatSelect = document.getElementById('format-select');
    formatSelect.addEventListener('change', function() {
        format_type = formatSelect.value;
    });

    
    /*
    const firstSetBtn = document.getElementById('firstSetBtn');
    const secondSetBtn = document.getElementById('secondSetBtn');
    const pagination1_1 = document.getElementById('pagination1_1');
    const pagination1_2 = document.getElementById('pagination1_2');
    const pagination2_1 = document.getElementById('pagination2_1');
    const pagination2_2 = document.getElementById('pagination2_2');
    const pagination2_3 = document.getElementById('pagination2_3');

    
    firstSetBtn.addEventListener('click', function() {
        classSet = 1;
        pagination1_1.style.display = 'block';
        pagination1_2.style.display = 'block';
        pagination2_1.style.display = 'none';
        pagination2_2.style.display = 'none';
        pagination2_3.style.display = 'none';
        firstSetBtn.style = "border: 3px solid red;"
        secondSetBtn.style.border = "none"
        
        const liItems = document.querySelectorAll('li');
        liItems.forEach(li => {
            li.classList.remove('ptype', 'active');
        });
        const class1ptype = document.getElementById('class1ptype');
        class1ptype.classList.add('ptype', 'active');
        ptype = 1;
    });

    
    secondSetBtn.addEventListener('click', function() {
        classSet = 2;
        pagination1_1.style.display = 'none';
        pagination1_2.style.display = 'none';
        pagination2_1.style.display = 'block';
        pagination2_2.style.display = 'block';
        pagination2_3.style.display = 'block';
        firstSetBtn.style.border = "none"
        secondSetBtn.style = "border: 3px solid red;"

        const liItems = document.querySelectorAll('li');
        liItems.forEach(li => {
            li.classList.remove('ptype', 'active');
        });
        const class2ptype = document.getElementById('class2ptype');
        class2ptype.classList.add('ptype', 'active');
        ptype = 1;
    });
    */ 
    const logoutBtn = document.getElementById('logout-button');
    logoutBtn.addEventListener('click', logout)
});
function logout() {
    localStorage.removeItem('username');
    localStorage.removeItem('login_set')
    localStorage.removeItem('login_mode')
    
    document.getElementById("form_container").style.display = "block";
    document.querySelector(".content").style.display = "none";
    document.querySelector("nav").style.display = "none";
    classSet = 1;
    mode = 'annotate'
    images = [];
    imagesName = [];
    currentImageIndex = 0;
    menuItemsCompleted = [];
    completedImageName = [];
    parentImagesName = [];
    labels = [];
    annotations = [];
    prev_index = -1;
    detections = []; 
    paddings = [];
    document.getElementById('image-counter').textContent = '圖片數量 0 / 0';
    document.getElementById('complete-counter').textContent = '完成標註數量 0 / 0';
    updateImageMenu(imagesName);
    const container = document.getElementById('image-container');
    const imageDisplay = document.getElementById('image_display');
    if (imageDisplay) {
        container.removeChild(imageDisplay);
    }
    const divs = container.getElementsByClassName('overlay-div');
    while (divs.length > 0) {
        divs[0].parentNode.removeChild(divs[0]);
    }
    document.getElementById('login-set').selectedIndex = 0;
    document.getElementById('login-mode').selectedIndex = 0;
}
function handleFileSelect(event) {
    const downloadStatus = document.getElementById('download-status')
    downloadStatus.textContent = '圖片上傳中'; 
    let files = Array.from(event.target.files);
    if (!files || files.length === 0) return;

    if (images.length >= 500) {
        alert("已達到圖片上限 無法繼續上傳，請先完成目前圖片標註！");
        return;
    }
    
    const promises = files.filter(f => f.type.match('image.*')).map(f => readFileAsDataURL(f));
    const imageContainer = document.getElementById('image-container');
    imageContainer.addEventListener('mouseover', imageMouseOverHandler);
    imageContainer.addEventListener('mouseout', imageMouseOutHandler);
    setTimeout(() => {
        downloadStatus.textContent = '已完成圖片上傳'; 
    }, 10000);
    setTimeout(() => {
        downloadStatus.textContent = ''; 
    }, 5000);
    
}

function readFileAsDataURL(file) {
    return new Promise((reject) => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                // avoid repeat uploading
                if (parentImagesName.indexOf(file.name.split('.').slice(0, -1).join('.')) === -1) {
                    const [newImages, newImageNames] = splitImage(img, file, split_size, mode);
                    images.push(...newImages);
                    imagesName.push(...newImageNames)
                    updateImageMenu(imagesName); 
                    showImage(currentImageIndex, change=false);
                } 
            };
            img.src = event.target.result;
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}

async function handleLabelSelect(event) {
    let files = Array.from(event.target.files);
    if (!files || files.length === 0) {
        console.error('未选择文件');
        return;
    }
    const promises = files.filter(f => f.type.match('text/plain')).map(f => readLabel(f));
}

async function readLabel(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/upload_yolo_labels', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            const labels_data = await response.json();
            if (labels_data !== null) {
                const index = imagesName.indexOf(labels_data[0].filename);

                if (index !== -1) {
                    anno_ids[index] = labels_data[0].anno_id
                    annotations[index] = labels_data[0].annos 
                    labels[index] = labels_data[0].labels
                    // showImage(currentImageIndex, modify=true);
                } else {
                    console.log(labels_data[0].filename, '.jpg not uploaded')
                }
            } else {
                console.log(labels_data[0].filename, '.txt has no label')
            }
        } else {
            console.error('上传失败:', response.statusText);
        }
    } catch (error) {
        console.error('上传时出错:', error);
    }
    showImage(currentImageIndex, modify=true);
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

    const parentImage = {
        name: file.name,
        width: image.width,
        height: image.height, 
        split_size: size
    }
    const childImages = [];
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
   
    const fileName = file.name.split('.').slice(0, -1).join('.');
    parentImagesName.push(fileName)
    
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
            detections.push([]);
            labels.push([])
            annotations.push([])
            anno_ids.push(0)
            const padding = createPadding(h, w, numH, numW, size, padH, padW)
            paddings.push([padding])
            child_img = {
                name: imageName,
                location: [h, w],
                paddings: padding
            };
            childImages.push(child_img)
        }
    }
    // if is modify mode 
    if (mode === 'modify') {
        if (image.height === 480  & image.width === 480) {
            return [[canvas.toDataURL()], [fileName]]
        } else {
            console.error("image size wrong")
        }   
    }
    add_parent_child_images(parentImage, childImages)
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

function showImage(index, change = true, modify=false) {
    const container = document.getElementById('image-container');
    const imageDisplay = document.getElementById('image_display');
    if (imageDisplay) {
        container.removeChild(imageDisplay);
    }
    
    
    const img = document.createElement('img');
    img.id = 'image_display'
    // const img = document.getElementById('image-display');
    img.src = images[index];
    container.appendChild(img);
    /*
    if (change & index !== prev_index) {
        updateAnnotations(index);
        prev_index = currentImageIndex;
    }
    if (modify) {
        
    }
    */
    updateAnnotations(index);
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
    updatCompleteCounter()
}

function updateAnnotations(index) {
    // 獲取指定索引的圖片的標註結果
    const currentImageAnnotations = annotations[index];
    if (!currentImageAnnotations) {
        return; // 如果沒有標註結果，則返回
    }

    // 移除上一個圖片的標註
    const imageContainer = document.getElementById('image-container');
    const divs = imageContainer.getElementsByClassName('overlay-div');
    while (divs.length > 0) {
        divs[0].parentNode.removeChild(divs[0]);
    }

    // 顯示當前圖片的標註
    currentImageAnnotations.forEach(annotation => {
        const { id, x, y, w, h, anno_id} = annotation;
        const rect = document.createElement('div');
        rect.className = 'overlay-div';
        rect.style.position = 'absolute';
        rect.style.left = `${x}px`;
        rect.style.top = `${y}px`;
        rect.style.width = `${w}px`;
        rect.style.height = `${h}px`;
        rect.style.backgroundColor = classColors[id];
        rect.style.opacity = '0.5';
        rect.dataset.anno_id = anno_id;
        imageContainer.appendChild(rect);

        rect.addEventListener('click', function () {
            const removedAnnoId = this.dataset.anno_id;
            labels[currentImageIndex] = labels[currentImageIndex].filter(label => parseInt(label.anno_id) !== parseInt(removedAnnoId));
            annotations[currentImageIndex] = annotations[currentImageIndex].filter(anno => parseInt(anno.anno_id) !== parseInt(removedAnnoId));
            this.remove();
        });
    });
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
    document.getElementById("image-menu").style.display = "block";
    const menu = document.getElementById('image-menu');
    menu.innerHTML = ''; // 清空菜单内容

    // 为每张图像创建菜单项
    if (imageNames.length !== 0) {
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
    
            const downloadButtonContainer = document.createElement('div'); // 创建一个新的容器元素
            downloadButtonContainer.style.display = 'inline-block'; // 设置容器为内联块级元素
    
            // download-button
            const downloadButton = document.createElement('button');
            downloadButton.textContent = '下載圖片';
            downloadButton.className = 'download-image-button';
    
            downloadButton.addEventListener('click', async function () {
                downloadImage(index)
            });
    
            downloadButtonContainer.appendChild(downloadButton); // 将 download button 放入容器内
            menuItem.appendChild(downloadButtonContainer); // 将容器放入菜单项内
    
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
}

// 标记图像为已完成
function markImageAsCompleted(index) {
    updatCompleteCounter()
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
        completedImageName = completedImageName.filter(name => name !== imagesName[index]) 
        images.splice(index, 1);
        imagesName.splice(index, 1);
        detections.splice(index, 1);
        paddings.splice(index, 1);
        updateImageMenu(imagesName);
        labels.splice(index, 1);
        annotations.splice(index, 1)
        anno_ids.splice(index, 1)
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
            labels.splice(index, 1);
            annotations.splice(index, 1)
            anno_ids.splice(index, 1)
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
    div.dataset.anno_id = anno_ids[currentImageIndex];
    document.getElementById('image-container').appendChild(div);
    annotations[currentImageIndex].push({
        id: ptype - 1,
        x: divLeft,
        y: divTop, 
        w: bboxWidth,
        h: bboxHeight,
        anno_id: anno_ids[currentImageIndex] 
    })
    labels[currentImageIndex].push({
        id: ptype-1,
        x: xcenter * ratio / split_size,
        y: ycenter * ratio / split_size,
        w: bboxWidth * ratio / split_size,
        h: bboxHeight * ratio / split_size,
        anno_id: anno_ids[currentImageIndex]
    });
    anno_ids[currentImageIndex] += 1

    div.addEventListener('click', function () {
        const removedAnnoId = this.dataset.anno_id;
        labels[currentImageIndex] = labels[currentImageIndex].filter(label => parseInt(label.anno_id) !== parseInt(removedAnnoId));
        annotations[currentImageIndex] = annotations[currentImageIndex].filter(anno => parseInt(anno.anno_id) !== parseInt(removedAnnoId));
        this.remove();        
    });
}

function updateImageCounter(index) {
    document.getElementById('image-counter').textContent = '圖片數量 ' + (index + 1) + ' / ' + images.length;
}

function updatCompleteCounter() {
    document.getElementById('complete-counter').textContent = '完成標註數量 ' + completedImageName.length + ' / ' + images.length;
}

// reset all labels
function labelreset() {
    completedImageName = completedImageName.filter(name => name !== imagesName[currentImageIndex]) 
    labels[currentImageIndex] = []
    const menuItem = document.getElementById(`menu-item-${currentImageIndex}`);
    if (menuItem) {
        menuItem.classList.remove('completed'); // 添加已完成样式
    }
    annotations[currentImageIndex] = []
    anno_ids[currentImageIndex] = 0
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

async function complete_label_image() {
    const downloadStatus = document.getElementById('download-status')
    if (images[currentImageIndex]) {
        const imageData = images[currentImageIndex]
        const imageName = imagesName[currentImageIndex]
        const label = labels[currentImageIndex]

        try {
            downloadStatus.textContent = `正在儲存 ${imageName} ...`;
            await fetch('website/save_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image_data: imageData,
                    image_name: imageName,
                    format_type: format_type,
                    class_set: classSet,
                    username: localStorage.getItem('username')
                })
            });
            await fetch('website/save_annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image_name: imageName, 
                    format_type: format_type,
                    yolo_labels: label,
                    img_size:  split_size,
                    class_set: classSet,
                    username: localStorage.getItem('username')
                })
            });
            await fetch('website/add_labels_db', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    username: localStorage.getItem('username'),
                    image_name: imageName, 
                    yolo_labels: label,
                    class_set: classSet
                })
            });
            downloadStatus.textContent = '標註成功!';
            setTimeout(() => {
                downloadStatus.textContent = ''; // 清除下载状态文本
            }, 10000);
        } catch (error) {
            console.error('error: ', error)
            downloadStatus.textContent = '';
        }
    } else {
        downloadStatus.textContent = '沒有圖片可以標註.';
    }
}

function add_parent_child_images(parentImage, childImages) {
    const data = {
        username: localStorage.getItem('username'),
        parent_image: parentImage,
        child_images: childImages
    }

    fetch('website/add_img_db', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .catch(error => {
        console.error('There was a problem with your fetch operation:', error);
    });
}

function download_labeled_images() {
    const downloadStatus = document.getElementById('download-status');
    if (completedImageName.length > 0) {
        const queryString = `?class_set=${encodeURIComponent(classSet)}&filenames=${encodeURIComponent(JSON.stringify(completedImageName))}&format_type=${encodeURIComponent(format_type)}`;
        fetch(`website/download_annotations${queryString}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response => {
            // 根据响应状态码检查下载状态
            if (response.ok) {
                // 创建一个 <a> 元素来触发下载
                const link = document.createElement('a');
                link.href = response.url;
                link.download = 'annotations.zip';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                downloadStatus.textContent = '下载失败，请重试.';
            }
        }).catch(error => {
            console.error('发送请求时出错:', error);
            downloadStatus.textContent = '下载失败，请重试.';
        });
    } else {
        downloadStatus.textContent = '没有可下载的图片.';
    }
}

function imageMouseOverHandler(event) {
    const labelSizeInput = document.getElementById('label-size');
    const divSize = parseInt(labelSizeInput.value);
    
    const rect = this.getBoundingClientRect();
    const mouseX = event.clientX - rect.left - divSize/2;
    const mouseY = event.clientY - rect.top - divSize/2;

    const overlayDiv = document.createElement('div');
    overlayDiv.className = 'overlay-div-preview';
    overlayDiv.style.position = 'absolute';
    overlayDiv.style.left = `${mouseX}px`;
    overlayDiv.style.top = `${mouseY}px`;
    overlayDiv.style.width = `${divSize}px`;
    overlayDiv.style.height = `${divSize}px`;
    overlayDiv.style.backgroundColor = classColors[ptype - 1];
    overlayDiv.style.opacity = '0.5';

    overlayDiv.style.pointerEvents = 'none';
    document.getElementById('image-container').appendChild(overlayDiv);

    function updateOverlayPosition(event) {
        const newX = event.clientX - rect.left - divSize/2;
        const newY = event.clientY - rect.top - divSize/2;
        overlayDiv.style.left = `${newX}px`;
        overlayDiv.style.top = `${newY}px`;
    }

    function removeOverlay() {
        overlayDiv.remove();
        document.removeEventListener('mousemove', updateOverlayPosition);
        document.removeEventListener('mouseout', removeOverlay);
    }

    document.addEventListener('mousemove', updateOverlayPosition);
    document.addEventListener('mouseout', removeOverlay);
}

function imageMouseOutHandler() {
    const overlayDiv = document.querySelector('.overlay-div-preview');
    if (overlayDiv) {
        overlayDiv.remove();
    }
}




window.addEventListener('beforeunload', function(event) {
    // 取消事件的默认动作，以便显示确认框
    event.preventDefault();
    // 标准中未定义文本的返回值，但大多数浏览器会显示一个默认文本
    event.returnValue = '';
    
    // 显示确认框
    const confirmationMessage = '是否确定要重新加载页面？';
    event.returnValue = confirmationMessage; // 兼容旧版浏览器
    return confirmationMessage;
});

async function downloadImage(index) {
    const menuItem = document.getElementById(`menu-item-${index}`);
    if (!menuItem) return;
    const imageData = images[index]
    const imageName = imagesName[index]
    const downloadImageName = imageName.replace(/\s+/g, '_');
    try {
        // 保存图片
        await fetch('website/save_image_for_download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                image_data: imageData,
                image_name: imageName
            })
        });

        // 下载图片
        const queryString = `?filenames=${encodeURIComponent(JSON.stringify(downloadImageName))}`;
        const response = await fetch(`/download_image${queryString}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            // 创建一个 <a> 元素来触发下载
            const link = document.createElement('a');
            link.href = url;
            link.download = downloadImageName + '.jpg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // 释放对象 URL
            window.URL.revokeObjectURL(url);
        } else {
            // 处理下载失败情况
            console.error('下载失败:', response.statusText);
        }
    } catch (error) {
        console.error('发送请求时出错:', error);
    }
}

async function openFolderDialog() {
    const options = {
        // 只允许选择文件夹
        type: 'openDirectory',
    };
    // 使用浏览器提供的API打开文件对话框
    const folderHandle = await window.showDirectoryPicker(options);
    // 返回用户选择的文件夹路径
    return folderHandle;
}