function createOverlayControls(overlayGroupname, overlayClassname, overlayImgEls, doc) {
    const controlContainer = doc.querySelector('#overlay-control-template').content.cloneNode(true).querySelector('.overlay-control');

    controlContainer.querySelector('.overlay-control-name').textContent = overlayClassname;

    const opacitySlider = controlContainer.querySelector('.opacity-slider');
    opacitySlider.classList.add(`opacity-slider-${overlayGroupname}`);
    overlayImgEls.forEach(el => el.style.opacity = opacitySlider.value);
    opacitySlider.oninput = () => {
        overlayImgEls.forEach(el => el.style.opacity = opacitySlider.value);
    };

    const colorSlider = controlContainer.querySelector('.color-slider');
    const randomHue = Math.floor(Math.random() * 361);
    colorSlider.value = randomHue;
    overlayImgEls.forEach(el => el.style.filter = `hue-rotate(${randomHue}deg)`);
    colorSlider.oninput = () => {
        overlayImgEls.forEach(el => el.style.filter = `hue-rotate(${colorSlider.value}deg)`);
    };

    return controlContainer;
}

function hideAllOverlays(doc, overlayGroup) {
    doc.querySelectorAll(`.overlay-${overlayGroup}`).forEach((overlay) => {
        overlay.style.opacity = 0;
    });

    doc.querySelectorAll(`.opacity-slider-${overlayGroup}`).forEach((slider) => {
        slider.value = 0;
    });
}

function createBaseImage(image2d, doc) {
    const imgEl = doc.querySelector('#base-image-template').content.cloneNode(true).querySelector('img');
    imgEl.src = image2d.img;
    return imgEl;
}

function sanitizeClassname(classname) {
    return classname.replaceAll(' ', '-')
}

function createOverlayImage(overlay, doc) {
    const overlayEl = doc.querySelector('#overlay-template').content.cloneNode(true).querySelector('img');
    overlayEl.src = overlay.data;
    overlayEl.classList.add(`overlay-classname-${sanitizeClassname(overlay.classname)}`);
    return overlayEl;
}

function createImage2D(image2d, doc, allOverlayGroups) {
    const figureEl = doc.querySelector('#image2d-template').content.cloneNode(true).querySelector('.main-figure');
    figureEl.title = `Hold ctrl and click to copy to clipboard: \n${image2d.desc ?? ''}`;
    const captionEl = figureEl.querySelector('.main-figure-caption');
    if (image2d.caption) {
        captionEl.innerHTML = image2d.caption;
    } else {
        captionEl.remove();
    }
    figureEl.onclick = (event) => {

        if (event.ctrlKey) {
            navigator.clipboard.writeText(image2d.desc ?? '');
            // Show a visual effect
            figureEl.querySelector('img').classList.add("image-highlighted")
            setTimeout(() => {
                figureEl.querySelector('img').classList.remove("image-highlighted")
            }, 500);
        } else {
            // Toggle visibility of base image
            const new_visibility = figureEl.querySelector('.main-image').style.visibility === 'hidden' ? 'visible' : 'hidden';
            figureEl.querySelector('.main-image').style.visibility = new_visibility;
        }
    };
    figureEl.prepend(createBaseImage(image2d, doc));

    image2d.overlay_groups.forEach((overlay_group) => {
        if (!allOverlayGroups.has(overlay_group.overlay_type)) {
            allOverlayGroups.set(overlay_group.overlay_type, new Map());
        }
        overlay_group.overlays.forEach((overlay) => {
            let overlayImg = createOverlayImage(overlay, doc);
            overlayImg.classList.add(`overlay-${sanitizeClassname(overlay_group.overlay_type)}`);
            figureEl.prepend(overlayImg);

            let overlayGroup = allOverlayGroups.get(overlay_group.overlay_type);
            if (!overlayGroup.has(overlay.classname)) {
                overlayGroup.set(overlay.classname, []);
            }
            overlayGroup.get(overlay.classname).push(overlayImg);
        })
    });
    return figureEl;
}

export default function (component) {
    const { setStateValue, parentElement, data } = component;

    // Reloading the component currently removes and re-adds everything.
    const allOverlayGroups = new Map();
    parentElement.querySelector('#hide-all-groups-btn').onclick = () => {
        allOverlayGroups.forEach((_, overlayGroup) => {
            hideAllOverlays(parentElement, sanitizeClassname(overlayGroup));
        });
    };
    parentElement.querySelector('#toggle-captions-checkbox').onchange = (event) => {
        parentElement.querySelectorAll('.main-figure-caption').forEach((captionEl) => {
            captionEl.style.display = event.target.checked ? 'block' : 'none';
        });
    }
    parentElement.querySelector('.image-gallery').innerHTML = '';
    parentElement.querySelector('.image-gallery').style.maxHeight = data.gallery_height;
    parentElement.querySelector('.overlay-controls').innerHTML = '';

    data.images.forEach((image2d) => {
        const image2dEl = createImage2D(image2d, parentElement, allOverlayGroups);
        parentElement.querySelector('.image-gallery').appendChild(image2dEl);
    });

    allOverlayGroups.forEach((overlayGroup, overlayType) => {
        const groupControl = parentElement.querySelector('#overlay-control-group-template').content.cloneNode(true).querySelector('div');
        groupControl.querySelector('.overlay-control-group-name').textContent = overlayType;
        groupControl.querySelector('.overlay-control-hide-all-button').onclick = () => {
            hideAllOverlays(parentElement, sanitizeClassname(overlayType));
        };
        parentElement.querySelector('.overlay-controls').appendChild(groupControl);

        overlayGroup.forEach((overlayImgEls, overlayClassname) => {
            groupControl.appendChild(
                createOverlayControls(sanitizeClassname(overlayType), overlayClassname, overlayImgEls, parentElement)
            );
        });
    });

    parentElement.querySelector('.gallery-image-size-slider').oninput = (event) => {
        const newSize = `repeat(auto-fit, minmax(${event.target.value}px, 1fr))`;

        parentElement.querySelector('.image-gallery').style.gridTemplateColumns = newSize;
    };
}