// smt_app/static/script/add_product.js
// Garante que tudo rode com o DOM pronto
document.addEventListener('DOMContentLoaded', () => {
  // --- Referências de elementos (todos existem no HTML agora) ---
  const imagePreview = document.getElementById('imagePreview');
  const canvas = document.getElementById('annotationCanvas');
  const ctx = canvas ? canvas.getContext('2d') : null;

  const fiducialModal = document.getElementById('fiducialModal');
  const componentModal = document.getElementById('componentModal');

  const compPackageSelect = document.getElementById('compPackage');

  // CORREÇÃO: antes pegava a DIV; agora pegamos o INPUT e também o container para show/hide
  const newCompPackageContainer = document.getElementById('newCompPackage');
  const newCompPackageInput = document.getElementById('newCompPackageInput');

  const compPresenceInput = document.getElementById('compPresenceThreshold');
  const compSsimInput = document.getElementById('compSsimThreshold');

  // Preview de template 0° do pacote (novo)
  const packageTemplatePreviewContainer = document.getElementById('packageTemplatePreviewContainer');
  const packageTemplatePreview = document.getElementById('packageTemplatePreview');

  // Fluxo de DEFINIÇÃO (bodyAdjustModal)
  const bodyAdjustModal = document.getElementById('bodyAdjustModal');
  const bodyDefinitionContainer = document.getElementById('bodyDefinitionContainer');
  const bodyDefinitionStatus = document.getElementById('bodyDefinitionStatus');
  const defineBodyBtn = document.getElementById('defineBodyBtn');
  const bodyDefinitionDone = document.getElementById('bodyDefinitionDone');
  const confirmBodyBtn = document.getElementById('confirmBodyBtn');
  const cancelBodyBtn = document.getElementById('cancelBodyBtn');
  const bodyCanvasContainer = document.getElementById('bodyCanvasContainer');
  const bodyAdjustLoading = document.getElementById('bodyAdjustLoading');
  const undoBodyRectBtn = document.getElementById('undoBodyRectBtn');

  // Fluxo de CONFIRMAÇÃO (bodyConfirmModal)
  const bodyConfirmModal = document.getElementById('bodyConfirmModal');
  const bodyConfirmationContainer = document.getElementById('bodyConfirmationContainer');
  const bodyConfirmationStatus = document.getElementById('bodyConfirmationStatus');
  const confirmBodyBtn_Flow = document.getElementById('confirmBodyBtn_Flow');
  const bodyConfirmationDone = document.getElementById('bodyConfirmationDone');
  const bodyConfirmTemplateImg = document.getElementById('bodyConfirmTemplateImg');
  const bodyConfirmCanvasContainer = document.getElementById('bodyConfirmCanvasContainer');
  const bodyConfirmLoading = document.getElementById('bodyConfirmLoading');
  const confirmBodyPositionBtn = document.getElementById('confirmBodyPositionBtn');
  const cancelBodyPositionBtn = document.getElementById('cancelBodyPositionBtn');

  const compIsPolarizedCheckbox = document.getElementById('compIsPolarized');

  // Polarity modal
  const polarityDefinitionContainer = document.getElementById('polarityDefinitionContainer');
  const definePolarityBtn = document.getElementById('definePolarityBtn');
  const polarityDefinitionDone = document.getElementById('polarityDefinitionDone');

  const polarityAdjustModal = document.getElementById('polarityAdjustModal');
  const polarityAdjustLoading = document.getElementById('polarityAdjustLoading');
  const polarityCanvasContainer = document.getElementById('polarityCanvasContainer');
  const confirmPolarityBtn = document.getElementById('confirmPolarityBtn');
  const cancelPolarityBtn = document.getElementById('cancelPolarityBtn');

  // Estado da polaridade atual (em coordenadas dentro da ROI do componente)

  let polarityCanvas = null;
  let polarityCtx = null;
  let polarityImg = null;
  let currentPolarityRect = null; // {x,y,width,height} na ROI
  let isDrawingPolarity = false;
  let polStartX = 0, polStartY = 0;
  let polScaleX = 1.0, polScaleY = 1.0;

  // Quando marcar/desmarcar o checkbox, mostra/oculta a área de definição
  compIsPolarizedCheckbox.addEventListener('change', () => {
    if (compIsPolarizedCheckbox.checked) {
      polarityDefinitionContainer.classList.remove('hidden');
    } else {
      polarityDefinitionContainer.classList.add('hidden');
      polarityDefinitionDone.classList.add('hidden');
      currentPolarityRect = null;
    }
  });


  // Estado
  let currentMode = null;
  let isDrawing = false;
  let startX, startY;
  let annotations = { fiducials: [], components: [] };
  let lastAnnotationType = null;
  let tempAnnotation = null;
  let currentComponentRect = null;
  let currentComponentROI_b64 = null;
  let finalBodyRects = [];

  // Armazena templates criados nesta sessão (antes de irem ao DB)
  // Agora inclui base_rotation (ângulo em que o template nasceu)
  let newPackageTemplates = {};

  // Ajuste (multi-região)
  let bodyCanvas, bodyCtx, bodyImg;
  let currentBodyRects = [];
  let bodyScaleX = 1.0;
  let bodyScaleY = 1.0;
  const handleSize = 8;
  let selectedRectIndex = -1;
  let selectedHandle = null;
  let dragStartX, dragStartY;

  // Confirmação
  let confirmCanvas, confirmCtx, confirmImg;
  let confirmScaleX = 1.0;
  let confirmScaleY = 1.0;

  // Pré-visualização da imagem
  const goldenInput = document.getElementById('golden');
  if (goldenInput) {
    goldenInput.addEventListener('change', previewImage);
  }

  function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      imagePreview.onload = () => {
        document.getElementById('imagePreviewContainer').classList.remove('hidden');
        canvas.width = imagePreview.offsetWidth;
        canvas.height = imagePreview.offsetHeight;
        drawAnnotations();
      };
    };
    reader.readAsDataURL(file);
  }

  function drawAnnotations() {
    if (!ctx) return;

    if (imagePreview.naturalWidth === 0 || imagePreview.naturalHeight === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / imagePreview.naturalWidth;
    const scaleY = canvas.height / imagePreview.naturalHeight;

    if (tempAnnotation && tempAnnotation.type === 'fiducial_box') {
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        tempAnnotation.x * scaleX,
        tempAnnotation.y * scaleY,
        tempAnnotation.width * scaleX,
        tempAnnotation.height * scaleY
      );
    }

    annotations.fiducials.forEach(f => {
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(f.x * scaleX, f.y * scaleY, f.r * scaleX, 0, 2 * Math.PI);
      ctx.stroke();
    });

    annotations.components.forEach(comp => {
      ctx.strokeStyle = 'rgba(0, 0, 255, 0.8)';
      ctx.lineWidth = 3;
      ctx.strokeRect(comp.x * scaleX, comp.y * scaleY, comp.width * scaleX, comp.height * scaleY);

      if (comp.final_body_rects) {
        ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
        comp.final_body_rects.forEach(rect => {
          const regionX = (comp.x + rect.x) * scaleX;
          const regionY = (comp.y + rect.y) * scaleY;
          ctx.fillRect(regionX, regionY, rect.width * scaleX, rect.height * scaleY);
        });
      }
    });
  }

  function getMousePos(evt) {
    const rect = canvas.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
  }

  canvas.addEventListener('mousedown', function (e) {
    if (!currentMode) {
      alert('Selecione um modo (Fiducial ou Componente) primeiro.');
      return;
    }
    const pos = getMousePos(e);
    isDrawing = true;
    startX = pos.x;
    startY = pos.y;
  });

  canvas.addEventListener('mousemove', function (e) {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    drawAnnotations();

    const x = Math.min(startX, pos.x);
    const y = Math.min(startY, pos.y);
    const width = Math.abs(pos.x - startX);
    const height = Math.abs(pos.y - startY);

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
  });

  canvas.addEventListener('mouseup', async function (e) {
    if (!isDrawing) return;
    isDrawing = false;

    if (imagePreview.naturalWidth === 0 || imagePreview.naturalHeight === 0) {
      alert('Por favor, carregue uma Imagem Golden primeiro.');
      return;
    }

    const pos = getMousePos(e);
    const x = Math.min(startX, pos.x);
    const y = Math.min(startY, pos.y);
    const width = Math.abs(pos.x - startX);
    const height = Math.abs(pos.y - startY);
    if (width < 5 || height < 5) {
      drawAnnotations();
      return;
    }
    const scaleX = imagePreview.naturalWidth / canvas.width;
    const scaleY = imagePreview.naturalHeight / canvas.height;
    const naturalRect = {
      x: Math.round(x * scaleX),
      y: Math.round(y * scaleY),
      width: Math.round(width * scaleX),
      height: Math.round(height * scaleY)
    };

    if (currentMode === 'fiducial') {
      tempAnnotation = { type: 'fiducial_box', ...naturalRect };
      drawAnnotations();
      await showFiducialModal(naturalRect);
    } else if (currentMode === 'component') {
      drawAnnotations();
      showComponentModal(naturalRect);
    }
  });

  // --- Modais ---
  async function showFiducialModal(rect) {
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = rect.width;
    croppedCanvas.height = rect.height;
    const croppedCtx = croppedCanvas.getContext('2d');
    croppedCtx.drawImage(
      imagePreview,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      0,
      0,
      rect.width,
      rect.height
    );
    const imageData = croppedCanvas.toDataURL('image/png');

    try {
      const response = await fetch('/find_fiducials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: imageData })
      });
      if (!response.ok) throw new Error('Falha no servidor ao detectar fiducial.');
      const result = await response.json();
      if (result.circles && result.circles.length > 0) {
        document.getElementById('modalDebugImage').src = 'data:image/png;base64,' + result.debug_image;
        fiducialModal.classList.remove('hidden');

        const handleConfirm = () => {
          const ring = result.circles[0];
          annotations.fiducials.push({
            name: `FID${annotations.fiducials.length + 1}`,
            x: rect.x + ring.x,
            y: rect.y + ring.y,
            r: ring.r
          });
          lastAnnotationType = { type: 'fiducial' };
          closeModal(true);
        };

        const closeModal = () => {
          fiducialModal.classList.add('hidden');
          tempAnnotation = null;
          document.getElementById('confirmFiducialBtn').onclick = null;
          document.getElementById('cancelFiducialBtn').onclick = null;
          drawAnnotations();
          updateAnnotationsList();
        };

        document.getElementById('confirmFiducialBtn').onclick = handleConfirm;
        document.getElementById('cancelFiducialBtn').onclick = () => closeModal(false);
      } else {
        alert('Nenhum fiducial encontrado na área selecionada. Tente novamente.');
        tempAnnotation = null;
        drawAnnotations();
      }
    } catch (error) {
      alert('Erro: ' + error.message);
      tempAnnotation = null;
      drawAnnotations();
    }
  }

  compPackageSelect.addEventListener('change', () => {
    const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex];

    bodyDefinitionContainer.classList.add('hidden');
    bodyConfirmationContainer.classList.add('hidden');

    // Atualiza preview do template 0° do pacote, se existir
    if (packageTemplatePreviewContainer && packageTemplatePreview) {
      let previewShown = false;
      if (selectedOption) {
        const bodyMatrixPath = selectedOption.dataset.bodyMatrix;
        if (bodyMatrixPath) {
          // Caminho relativo ao /static
          packageTemplatePreview.src = `/static/${bodyMatrixPath}`;
          packageTemplatePreviewContainer.classList.remove('hidden');
          previewShown = true;
        }
      }
      if (!previewShown) {
        packageTemplatePreview.src = '';
        packageTemplatePreviewContainer.classList.add('hidden');
      }
    }

    if (compPackageSelect.value === '--new--') {
      newCompPackageContainer.classList.remove('hidden');
      compPresenceInput.value = 0.35;
      compSsimInput.value = 0.60;

      bodyDefinitionContainer.classList.remove('hidden');
      bodyDefinitionStatus.textContent = 'Pacote novo. O corpo DEVE ser definido.';
      bodyDefinitionDone.classList.add('hidden');
      defineBodyBtn.classList.remove('hidden');
      finalBodyRects = [];
    } else {
      newCompPackageContainer.classList.add('hidden');

      if (selectedOption && selectedOption.dataset.presence) {
        compPresenceInput.value = parseFloat(selectedOption.dataset.presence).toFixed(2);
        compSsimInput.value = parseFloat(selectedOption.dataset.ssim).toFixed(2);
      }

      const isBodyDefined = selectedOption && selectedOption.dataset.bodyDefined === 'true';
      const isRoiSizeDefined = selectedOption ? parseInt(selectedOption.dataset.templateRoiWidth || '0') > 0 : false;
      const isDefinedInSession = selectedOption ? !!newPackageTemplates[selectedOption.value] : false;

      if ((isBodyDefined && isRoiSizeDefined) || isDefinedInSession) {
        bodyConfirmationContainer.classList.remove('hidden');
        bodyConfirmationDone.classList.add('hidden');
        confirmBodyBtn_Flow.classList.remove('hidden');
        finalBodyRects = [];
      } else if (compPackageSelect.value) {
        bodyDefinitionContainer.classList.remove('hidden');
        bodyDefinitionStatus.textContent = 'Este pacote não tem corpo ou tamanho de ROI. O corpo DEVE ser definido.';
        bodyDefinitionDone.classList.add('hidden');
        defineBodyBtn.classList.remove('hidden');
        finalBodyRects = [];
      }
    }
  });

  function updatePackageDropdown(packageName, width, height) {
    let optionExists = false;
    for (let i = 0; i < compPackageSelect.options.length; i++) {
      if (compPackageSelect.options[i].value === packageName) {
        optionExists = true;
        const opt = compPackageSelect.options[i];
        opt.dataset.bodyDefined = 'true';
        opt.dataset.templateRoiWidth = width;
        opt.dataset.templateRoiHeight = height;
        // body_matrix virá do backend somente depois de salvo; aqui deixamos vazio
        if (!opt.dataset.bodyMatrix) {
          opt.dataset.bodyMatrix = '';
        }
        break;
      }
    }
    if (!optionExists) {
      const newOption = document.createElement('option');
      newOption.value = packageName;
      newOption.text = packageName;
      newOption.dataset.bodyDefined = 'true';
      newOption.dataset.presence = '0.35';
      newOption.dataset.ssim = '0.60';
      newOption.dataset.templateRoiWidth = width;
      newOption.dataset.templateRoiHeight = height;
      newOption.dataset.bodyMatrix = '';

      const newOptionEntry = compPackageSelect.querySelector('option[value="--new--"]');
      compPackageSelect.insertBefore(newOption, newOptionEntry);
    }
  }

  function pushComponentAnnotation(packageName, isDefining) {
    if (!currentComponentRect) {
      alert('Nenhum componente selecionado.');
      return;
    }
  
    const nameInput = document.getElementById('compName');
    const name = nameInput.value.trim() || `COMP${annotations.components.length + 1}`;
  
    const rotationVal = parseInt(document.getElementById('compRotation').value) || 0;
    const isPolarized = !!compIsPolarizedCheckbox.checked;
  
    if (isPolarized && !currentPolarityRect) {
      alert('Você marcou o componente como polarizado, mas não definiu o BOX de polaridade.');
      return;
    }
  
    const newComponent = {
      name,
      package: packageName,
      rotation: rotationVal,
      x: currentComponentRect.x,
      y: currentComponentRect.y,
      width: currentComponentRect.width,
      height: currentComponentRect.height,
      is_polarized: isPolarized,
      polarity_rect: isPolarized && currentPolarityRect ? {
        x: currentPolarityRect.x,
        y: currentPolarityRect.y,
        width: currentPolarityRect.width,
        height: currentPolarityRect.height
      } : null
    };
  
    if (isDefining) {
      newComponent.final_body_rects = finalBodyRects;
      newComponent.component_roi_b64 = currentComponentROI_b64;
  
      updatePackageDropdown(packageName, currentComponentRect.width, currentComponentRect.height);
  
      newPackageTemplates[packageName] = {
        roi_b64: currentComponentROI_b64,
        body_rects: finalBodyRects,
        roi_width: currentComponentRect.width,
        roi_height: currentComponentRect.height,
        base_rotation: rotationVal
      };
    }
  
    annotations.components.push(newComponent);
    lastAnnotationType = { type: 'component' };
  
    // Reset do modal
    nameInput.value = '';
    document.getElementById('compRotation').value = '0';
    compIsPolarizedCheckbox.checked = false;
    currentPolarityRect = null;
    polarityDefinitionContainer.classList.add('hidden');
    polarityDefinitionDone.classList.add('hidden');
  
    componentModal.classList.add('hidden');
  }
  
  

  function showComponentModal(rect) {
    document.getElementById('compName').value = `COMP${annotations.components.length + 1}`;
    compPackageSelect.value = '';
    newCompPackageInput.value = '';
    newCompPackageContainer.classList.add('hidden');
    document.getElementById('compRotation').value = '0';
    compPresenceInput.value = '';
    compSsimInput.value = '';
    bodyDefinitionContainer.classList.add('hidden');
    bodyConfirmationContainer.classList.add('hidden');
    bodyDefinitionDone.classList.add('hidden');
    bodyConfirmationDone.classList.add('hidden');
  
    // reset polaridade
    compIsPolarizedCheckbox.checked = false;
    currentPolarityRect = null;
    polarityDefinitionContainer.classList.add('hidden');
    polarityDefinitionDone.classList.add('hidden');
  
    // Esconde preview do template ao abrir o modal
    if (packageTemplatePreviewContainer && packageTemplatePreview) {
      packageTemplatePreview.src = '';
      packageTemplatePreviewContainer.classList.add('hidden');
    }
  
    currentComponentRect = rect;
    finalBodyRects = [];
  
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = rect.width;
    croppedCanvas.height = rect.height;
    const croppedCtx = croppedCanvas.getContext('2d');
    croppedCtx.drawImage(
      imagePreview,
      rect.x, rect.y, rect.width, rect.height,
      0, 0, rect.width, rect.height
    );
    currentComponentROI_b64 = croppedCanvas.toDataURL('image/png');
  
    componentModal.classList.remove('hidden');

    const handleSave = () => {
      let packageName = compPackageSelect.value;
      if (packageName === '--new--') {
        packageName = newCompPackageInput.value.trim();
        if (!packageName) {
          alert('Por favor, insira um nome para o novo pacote.');
          return;
        }
      }
      if (!packageName) {
        alert('Selecione ou cadastre um pacote.');
        return;
      }

      const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex] || {};
      const isBodyDefined = selectedOption.dataset ? selectedOption.dataset.bodyDefined === 'true' : false;
      const isRoiSizeDefined = selectedOption.dataset ? parseInt(selectedOption.dataset.templateRoiWidth || '0') > 0 : false;
      const isDefinedInSession = !!newPackageTemplates[packageName];

      if ((!isBodyDefined || !isRoiSizeDefined) && !isDefinedInSession) {
        if (finalBodyRects.length === 0) {
          alert("Este pacote é novo ou não tem um corpo/tamanho definidos. Por favor, clique em 'Sugerir & Definir Regiões' e defina pelo menos uma região.");
          return;
        }
        pushComponentAnnotation(packageName, true);
        closeModal();
      } else {
        if (bodyConfirmationDone.classList.contains('hidden')) {
          alert("Este pacote já existe. Por favor, clique em 'Encontrar & Confirmar Corpo' para ajustar a ROI.");
          return;
        }
        pushComponentAnnotation(packageName, false);
        closeModal();
      }
    };

    const closeModal = () => {
      componentModal.classList.add('hidden');
      document.getElementById('saveComponentBtn').onclick = null;
      document.getElementById('cancelComponentBtn').onclick = null;
      currentComponentRect = null;
      currentComponentROI_b64 = null;
      finalBodyRects = [];
      drawAnnotations();
      updateAnnotationsList();
    };

    document.getElementById('saveComponentBtn').onclick = handleSave;
    document.getElementById('cancelComponentBtn').onclick = closeModal;
  }

  // --- Fluxo DEFINIÇÃO ---
  defineBodyBtn.addEventListener('click', async () => {
    if (!currentComponentROI_b64) {
      alert('Erro: ROI do componente não encontrada.');
      return;
    }
    defineBodyBtn.disabled = true;
    defineBodyBtn.textContent = 'Sugerindo...';
    try {
      const response = await fetch('/suggest_body', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ component_roi_b64: currentComponentROI_b64 })
      });
      const suggestionRects = await response.json();
      if (!response.ok) throw new Error(suggestionRects.error);
      openBodyAdjustModal(currentComponentROI_b64, suggestionRects);
    } catch (error) {
      alert('Erro ao sugerir corpo: ' + error.message);
    } finally {
      defineBodyBtn.disabled = false;
      defineBodyBtn.textContent = '1. Sugerir & Definir Regiões';
    }
  });

  definePolarityBtn.addEventListener('click', () => {
    if (!currentComponentROI_b64) {
      alert('Erro: ROI do componente não encontrada.');
      return;
    }
    openPolarityAdjustModal(currentComponentROI_b64);
  });

  function openPolarityAdjustModal(roi_b64) {
    polarityAdjustLoading.style.display = 'block';
    polarityCanvasContainer.innerHTML = '';
    polarityAdjustModal.classList.remove('hidden');

    isDrawingPolarity = false;

    polarityImg = new Image();
    polarityImg.src = roi_b64;
    polarityImg.onload = () => {
      polarityCanvas = document.createElement('canvas');
      polarityCtx = polarityCanvas.getContext('2d');

      const containerWidth = polarityCanvasContainer.offsetWidth || polarityImg.width;
      const containerHeight = (polarityImg.height / polarityImg.width) * containerWidth;

      polarityCanvas.width = containerWidth;
      polarityCanvas.height = containerHeight;

      polScaleX = polarityImg.width / containerWidth;
      polScaleY = polarityImg.height / containerHeight;

      polarityCanvasContainer.appendChild(polarityCanvas);

      drawPolarityCanvas();

      polarityCanvas.addEventListener('mousedown', polarityMouseDown);
      polarityCanvas.addEventListener('mousemove', polarityMouseMove);
      polarityCanvas.addEventListener('mouseup', polarityMouseUp);
      polarityCanvas.addEventListener('mouseleave', polarityMouseUp);

      polarityAdjustLoading.style.display = 'none';
    };
  }

  function drawPolarityCanvas() {
    if (!polarityCtx || !polarityImg) return;
    polarityCtx.clearRect(0, 0, polarityCanvas.width, polarityCanvas.height);
    polarityCtx.drawImage(polarityImg, 0, 0, polarityCanvas.width, polarityCanvas.height);

    if (currentPolarityRect) {
      polarityCtx.save();
      polarityCtx.strokeStyle = '#10b981'; // verde
      polarityCtx.lineWidth = 2;
      const vx = currentPolarityRect.x / polScaleX;
      const vy = currentPolarityRect.y / polScaleY;
      const vw = currentPolarityRect.width / polScaleX;
      const vh = currentPolarityRect.height / polScaleY;
      polarityCtx.strokeRect(vx, vy, vw, vh);
      polarityCtx.restore();
    }
  }

  function polarityMouseDown(e) {
    const rect = polarityCanvas.getBoundingClientRect();
    polStartX = e.clientX - rect.left;
    polStartY = e.clientY - rect.top;
    isDrawingPolarity = true;
    currentPolarityRect = null;
  }

  function polarityMouseMove(e) {
    if (!isDrawingPolarity) return;
    const rect = polarityCanvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    const x = Math.min(polStartX, currentX);
    const y = Math.min(polStartY, currentY);
    const w = Math.abs(currentX - polStartX);
    const h = Math.abs(currentY - polStartY);

    currentPolarityRect = {
      x: Math.round(x * polScaleX),
      y: Math.round(y * polScaleY),
      width: Math.max(1, Math.round(w * polScaleX)),
      height: Math.max(1, Math.round(h * polScaleY))
    };

    drawPolarityCanvas();
  }

  function polarityMouseUp() {
    if (!isDrawingPolarity) return;
    isDrawingPolarity = false;
    drawPolarityCanvas();
  }

  confirmPolarityBtn.addEventListener('click', () => {
    if (!currentPolarityRect) {
      alert('Desenhe um retângulo de polaridade antes de confirmar.');
      return;
    }
    polarityDefinitionDone.classList.remove('hidden');
    closePolarityAdjustModal();
  });

  function closePolarityAdjustModal() {
    polarityAdjustModal.classList.add('hidden');
    if (polarityCanvas) {
      polarityCanvas.removeEventListener('mousedown', polarityMouseDown);
      polarityCanvas.removeEventListener('mousemove', polarityMouseMove);
      polarityCanvas.removeEventListener('mouseup', polarityMouseUp);
      polarityCanvas.removeEventListener('mouseleave', polarityMouseUp);
    }
  }

  cancelPolarityBtn.addEventListener('click', () => {
    closePolarityAdjustModal();
  });


  function openBodyAdjustModal(roi_b64, suggestionRects) {
    bodyAdjustLoading.style.display = 'block';
    bodyCanvasContainer.innerHTML = '';
    bodyAdjustModal.classList.remove('hidden');

    currentBodyRects = suggestionRects || [];
    selectedRectIndex = -1;
    selectedHandle = null;
    isDrawing = false;

    bodyImg = new Image();
    bodyImg.src = roi_b64;
    bodyImg.onload = () => {
      bodyCanvas = document.createElement('canvas');
      bodyCtx = bodyCanvas.getContext('2d');

      const containerWidth = bodyCanvasContainer.offsetWidth;
      const aspect = bodyImg.naturalHeight / bodyImg.naturalWidth;
      const canvasWidth = containerWidth;
      const canvasHeight = containerWidth * aspect;

      bodyCanvas.width = canvasWidth;
      bodyCanvas.height = canvasHeight;
      bodyCanvasContainer.style.height = canvasHeight + 'px';

      bodyCanvasContainer.appendChild(bodyCanvas);
      bodyAdjustLoading.style.display = 'none';

      bodyScaleX = canvasWidth / bodyImg.naturalWidth;
      bodyScaleY = canvasHeight / bodyImg.naturalHeight;

      bodyCanvas.onmousedown = onBodyCanvasMouseDown;
      bodyCanvas.onmousemove = onBodyCanvasMouseMove;
      bodyCanvas.onmouseup = onBodyCanvasMouseUp;
      bodyCanvas.onmouseout = onBodyCanvasMouseUp;

      drawBodyCanvas();
    };
  }

  function drawBodyCanvas() {
    if (!bodyCanvas) return;
    bodyCtx.clearRect(0, 0, bodyCanvas.width, bodyCanvas.height);
    bodyCtx.drawImage(bodyImg, 0, 0, bodyCanvas.width, bodyCanvas.height);

    currentBodyRects.forEach((rect, index) => {
      const rx = rect.x * bodyScaleX;
      const ry = rect.y * bodyScaleY;
      const rw = rect.width * bodyScaleX;
      const rh = rect.height * bodyScaleY;

      if (index === selectedRectIndex) {
        bodyCtx.strokeStyle = 'rgba(255, 0, 0, 1.0)';
        bodyCtx.fillStyle = 'rgba(255, 0, 0, 0.4)';
        bodyCtx.lineWidth = 2;
        bodyCtx.strokeRect(rx, ry, rw, rh);
        bodyCtx.fillRect(rx, ry, rw, rh);

        bodyCtx.fillStyle = 'rgba(255, 0, 0, 1.0)';
        const handles = getResizeHandles(rect, bodyScaleX, bodyScaleY);
        for (const handle in handles) {
          const h = handles[handle];
          bodyCtx.fillRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
        }
      } else {
        bodyCtx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
        bodyCtx.fillStyle = 'rgba(0, 255, 0, 0.3)';
        bodyCtx.lineWidth = 2;
        bodyCtx.strokeRect(rx, ry, rw, rh);
        bodyCtx.fillRect(rx, ry, rw, rh);
      }
    });
  }

  function getResizeHandles(rect, scaleX, scaleY) {
    const rx = rect.x * scaleX;
    const ry = rect.y * scaleY;
    const rw = rect.width * scaleX;
    const rh = rect.height * scaleY;
    return {
      tl: { x: rx, y: ry },
      tr: { x: rx + rw, y: ry },
      bl: { x: rx, y: ry + rh },
      br: { x: rx + rw, y: ry + rh }
    };
  }

  function getMousePosOnBodyCanvas(e) {
    const rect = bodyCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y, imgX: x / bodyScaleX, imgY: y / bodyScaleY };
  }

  function onBodyCanvasMouseDown(e) {
    const pos = getMousePosOnBodyCanvas(e);
    selectedHandle = null;
    selectedRectIndex = -1;

    for (let i = currentBodyRects.length - 1; i >= 0; i--) {
      const rect = currentBodyRects[i];
      const handles = getResizeHandles(rect, bodyScaleX, bodyScaleY);
      for (const handleName in handles) {
        const h = handles[handleName];
        if (Math.abs(pos.x - h.x) < handleSize && Math.abs(pos.y - h.y) < handleSize) {
          selectedHandle = handleName;
          selectedRectIndex = i;
          dragStartX = pos.x;
          dragStartY = pos.y;
          drawBodyCanvas();
          return;
        }
      }
    }

    for (let i = currentBodyRects.length - 1; i >= 0; i--) {
      const rect = currentBodyRects[i];
      const rx = rect.x * bodyScaleX;
      const ry = rect.y * bodyScaleY;
      const rw = rect.width * bodyScaleX;
      const rh = rect.height * bodyScaleY;

      if (pos.x > rx && pos.x < rx + rw && pos.y > ry && pos.y < ry + rh) {
        selectedHandle = 'm';
        selectedRectIndex = i;
        dragStartX = pos.imgX;
        dragStartY = pos.imgY;
        drawBodyCanvas();
        return;
      }
    }

    isDrawing = true;
    startX = pos.imgX;
    startY = pos.imgY;
  }

  function onBodyCanvasMouseMove(e) {
    const pos = getMousePosOnBodyCanvas(e);

    if (selectedHandle && selectedHandle !== 'm') {
      const rect = currentBodyRects[selectedRectIndex];
      const dx = (pos.x - dragStartX) / bodyScaleX;
      const dy = (pos.y - dragStartY) / bodyScaleY;

      if (selectedHandle.includes('t')) { rect.y += dy; rect.height -= dy; }
      if (selectedHandle.includes('l')) { rect.x += dx; rect.width -= dx; }
      if (selectedHandle.includes('b')) { rect.height += dy; }
      if (selectedHandle.includes('r')) { rect.width += dx; }

      dragStartX = pos.x;
      dragStartY = pos.y;
      drawBodyCanvas();
    } else if (selectedHandle === 'm') {
      const rect = currentBodyRects[selectedRectIndex];
      const dx = pos.imgX - dragStartX;
      const dy = pos.imgY - dragStartY;

      rect.x += dx;
      rect.y += dy;

      dragStartX = pos.imgX;
      dragStartY = pos.imgY;
      drawBodyCanvas();
    } else if (isDrawing) {
      drawBodyCanvas();

      const x = Math.min(startX, pos.imgX) * bodyScaleX;
      const y = Math.min(startY, pos.imgY) * bodyScaleY;
      const w = Math.abs(pos.imgX - startX) * bodyScaleX;
      const h = Math.abs(pos.imgY - startY) * bodyScaleY;

      bodyCtx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
      bodyCtx.lineWidth = 1;
      bodyCtx.strokeRect(x, y, w, h);
    }
  }

  function onBodyCanvasMouseUp(e) {
    if (isDrawing) {
      isDrawing = false;
      const pos = getMousePosOnBodyCanvas(e);
      const currentX = pos.imgX;
      const currentY = pos.imgY;

      const width = Math.abs(currentX - startX);
      const height = Math.abs(currentY - startY);

      if (width > 2 && height > 2) {
        currentBodyRects.push({
          x: Math.round(Math.min(startX, currentX)),
          y: Math.round(Math.min(startY, currentY)),
          width: Math.round(width),
          height: Math.round(height)
        });
        selectedRectIndex = currentBodyRects.length - 1;
      }
    } else if (selectedHandle && selectedHandle !== 'm') {
      const rect = currentBodyRects[selectedRectIndex];
      if (rect.width < 0) {
        rect.x += rect.width;
        rect.width *= -1;
      }
      if (rect.height < 0) {
        rect.y += rect.height;
        rect.height *= -1;
      }
    }

    selectedHandle = null;
    drawBodyCanvas();
  }

  undoBodyRectBtn.addEventListener('click', () => {
    if (selectedRectIndex !== -1) {
      currentBodyRects.splice(selectedRectIndex, 1);
      selectedRectIndex = -1;
    } else if (currentBodyRects.length > 0) {
      currentBodyRects.pop();
    }
    drawBodyCanvas();
  });

  confirmBodyBtn.addEventListener('click', () => {
    if (currentBodyRects.length === 0) {
      alert('Defina pelo menos uma região de corpo antes de confirmar.');
      return;
    }
  
    // Copia estática das regiões de corpo
    finalBodyRects = currentBodyRects.map(r => ({
      x: Math.round(r.x),
      y: Math.round(r.y),
      width: Math.round(r.width),
      height: Math.round(r.height),
    }));
  
    // Marca visualmente que o corpo foi definido
    bodyDefinitionDone.classList.remove('hidden');
    bodyAdjustModal.classList.add('hidden');
  
    // --- NOVO: se o componente é polarizado, já abre o fluxo de BOX de polaridade ---
    if (compIsPolarizedCheckbox.checked) {
      // exibe o container de polaridade
      polarityDefinitionContainer.classList.remove('hidden');
      // força o usuário a desenhar novamente, caso já tivesse algo antigo
      polarityDefinitionDone.classList.add('hidden');
  
      if (!currentComponentROI_b64) {
        alert('Erro: ROI do componente não encontrada para definir a polaridade.');
        return;
      }
  
      // Abre o modal para desenhar o BOX de polaridade
      openPolarityAdjustModal(currentComponentROI_b64);
    }
  });
  

  cancelBodyBtn.addEventListener('click', () => {
    bodyAdjustModal.classList.add('hidden');
    bodyCanvasContainer.innerHTML = '';
    bodyCanvasContainer.style.height = 'auto';
    bodyCanvas = null;
    currentBodyRects = [];
    selectedRectIndex = -1;
  });

  // --- Fluxo CONFIRMAÇÃO ---
  confirmBodyBtn_Flow.addEventListener('click', async () => {
    const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex];
    const packageName = selectedOption ? selectedOption.value : '';
    const rotation = parseInt(document.getElementById('compRotation').value) || 0;

    if (!packageName) {
      alert('Erro: Pacote não selecionado.');
      return;
    }

    confirmBodyBtn_Flow.disabled = true;
    confirmBodyBtn_Flow.textContent = 'Procurando...';
    bodyConfirmLoading.style.display = 'block';
    bodyConfirmCanvasContainer.innerHTML = '';
    bodyConfirmTemplateImg.src = '';

    let template_roi_size;
    let url = '';
    let payload = {};
    let effectiveRotation = rotation; // default

    if (newPackageTemplates[packageName]) {
      // Template criado nesta sessão (pode ter nascido já rotacionado)
      const templateData = newPackageTemplates[packageName];

      // Compensa a rotação base em que o template foi criado
      const baseRot = parseInt(templateData.base_rotation || 0) || 0;
      effectiveRotation = ((rotation - baseRot) % 360 + 360) % 360;

      url = '/find_body_in_roi_with_template';
      payload = {
        component_roi_b64: currentComponentROI_b64,
        template_roi_b64: templateData.roi_b64,
        template_body_rects: templateData.body_rects,
        rotation: effectiveRotation
      };
      template_roi_size = { width: templateData.roi_width, height: templateData.roi_height };
    } else {
      // Template vindo do DB (assumimos base 0°)
      url = '/find_body_in_roi';
      payload = { component_roi_b64: currentComponentROI_b64, package_name: packageName, rotation };
      template_roi_size = {
        width: parseInt(selectedOption.dataset.templateRoiWidth),
        height: parseInt(selectedOption.dataset.templateRoiHeight)
      };
    }

    if (!template_roi_size || template_roi_size.width === 0) {
      alert('Erro: Tamanho da ROI para este pacote é desconhecido. Defina o pacote primeiro.');
      confirmBodyBtn_Flow.disabled = false;
      confirmBodyBtn_Flow.textContent = '1. Encontrar & Confirmar Corpo';
      return;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);

      openBodyConfirmModal(result.body_rect, result.template_b64, template_roi_size, effectiveRotation);
    } catch (error) {
      alert('Erro ao encontrar corpo: ' + error.message);
    } finally {
      confirmBodyBtn_Flow.disabled = false;
      confirmBodyBtn_Flow.textContent = '1. Encontrar & Confirmar Corpo';
    }
  });

  function openBodyConfirmModal(found_body_rect, template_b64, template_roi_size, effectiveRotation) {
    bodyConfirmModal.classList.remove('hidden');
    bodyConfirmLoading.style.display = 'block';
    bodyConfirmCanvasContainer.innerHTML = '';
    bodyConfirmTemplateImg.src = 'data:image/png;base64,' + template_b64;

    confirmImg = new Image();
    confirmImg.src = currentComponentROI_b64;
    confirmImg.onload = () => {
      confirmCanvas = document.createElement('canvas');
      confirmCtx = confirmCanvas.getContext('2d');

      const containerWidth = bodyConfirmCanvasContainer.offsetWidth;
      const aspect = confirmImg.naturalHeight / confirmImg.naturalWidth;
      const canvasWidth = containerWidth;
      const canvasHeight = containerWidth * aspect;

      confirmCanvas.width = canvasWidth;
      confirmCanvas.height = canvasHeight;
      bodyConfirmCanvasContainer.style.height = canvasHeight + 'px';
      bodyConfirmCanvasContainer.appendChild(confirmCanvas);

      confirmScaleX = canvasWidth / confirmImg.naturalWidth;
      confirmScaleY = canvasHeight / confirmImg.naturalHeight;

      confirmCtx.drawImage(confirmImg, 0, 0, canvasWidth, canvasHeight);

      confirmCtx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
      confirmCtx.lineWidth = 2;
      confirmCtx.strokeRect(
        found_body_rect.x * confirmScaleX,
        found_body_rect.y * confirmScaleY,
        found_body_rect.width * confirmScaleX,
        found_body_rect.height * confirmScaleY
      );

      bodyConfirmLoading.style.display = 'none';
    };

    confirmBodyPositionBtn.onclick = () => {
      const body_center_x = found_body_rect.x + found_body_rect.width / 2;
      const body_center_y = found_body_rect.y + found_body_rect.height / 2;

      const abs_center_x = currentComponentRect.x + body_center_x;
      const abs_center_y = currentComponentRect.y + body_center_y;

      // Se a rotação efetiva for 90° ou 270°, a ROI do template “vira”, logo trocamos LxA
      const needsSwap = (effectiveRotation % 180) !== 0;
      const roiW = needsSwap ? template_roi_size.height : template_roi_size.width;
      const roiH = needsSwap ? template_roi_size.width : template_roi_size.height;

      const final_roi_x = abs_center_x - roiW / 2;
      const final_roi_y = abs_center_y - roiH / 2;

      currentComponentRect = {
        x: Math.round(final_roi_x),
        y: Math.round(final_roi_y),
        width: roiW,
        height: roiH
      };

      currentComponentROI_b64 = null;
      finalBodyRects = [];

      bodyConfirmModal.classList.add('hidden');
      bodyConfirmationDone.classList.remove('hidden');
      confirmBodyBtn_Flow.classList.add('hidden');
    };

    cancelBodyPositionBtn.onclick = () => {
      bodyConfirmModal.classList.add('hidden');
    };
  }

  // Controles gerais
  function setActiveMode(mode, instructionsText) {
    currentMode = mode;
    ['fiducialModeBtn', 'componentModeBtn'].forEach(id => {
      const el = document.getElementById(id);
      el.classList.remove('bg-indigo-600', 'text-white');
      el.classList.add('btn-secondary');
    });
    const btn = document.getElementById(`${mode}ModeBtn`);
    btn.classList.remove('btn-secondary');
    btn.classList.add('bg-indigo-600', 'text-white');
    document.getElementById('instructions').textContent = instructionsText;
  }

  document.getElementById('fiducialModeBtn').addEventListener('click', () =>
    setActiveMode('fiducial', 'Arraste na imagem para marcar a área de um fiducial.')
  );
  document.getElementById('componentModeBtn').addEventListener('click', () =>
    setActiveMode('component', 'Arraste para marcar a área de um componente (caixa azul).')
  );

  document.getElementById('undoBtn').addEventListener('click', () => {
    if (!lastAnnotationType) return;
    if (lastAnnotationType.type === 'fiducial') {
      annotations.fiducials.pop();
    } else if (lastAnnotationType.type === 'component') {
      const removedComp = annotations.components.pop();

      if (removedComp && removedComp.final_body_rects && newPackageTemplates[removedComp.package]) {
        const stillInUse = annotations.components.some(c => c.package === removedComp.package);
        if (!stillInUse) {
          delete newPackageTemplates[removedComp.package];

          const opt = compPackageSelect.querySelector(`option[value="${removedComp.package}"]`);
          if (opt && !opt.dataset.presence) {
            opt.remove();
          } else if (opt) {
            opt.dataset.templateRoiWidth = '0';
            opt.dataset.templateRoiHeight = '0';
          }
        }
      }
    }
    lastAnnotationType = null;
    drawAnnotations();
    updateAnnotationsList();
  });

  function updateAnnotationsList() {
    const fiducialsList = document.getElementById('fiducialsList');
    const componentsList = document.getElementById('componentsList');
    fiducialsList.innerHTML = '';
    componentsList.innerHTML = '';
    annotations.fiducials.forEach((f, i) => {
      fiducialsList.innerHTML += `<li class="text-sm">Fiducial ${i + 1}: (x:${f.x}, y:${f.y}, r:${f.r})</li>`;
    });
    annotations.components.forEach((c) => {
      let bodyInfo =
        c.final_body_rects && c.final_body_rects.length > 0
          ? 'com corpo definido'
          : 'usando corpo do pacote';
      if (!c.package) bodyInfo = 'sem pacote';
      componentsList.innerHTML += `<li class="text-sm">${c.name} (${c.package}, ${c.rotation}°): ${bodyInfo}</li>`;
    });
    const polText = comp.is_polarized ? ' (polarizado)' : '';
    li.textContent = `${comp.name} [${comp.package}]${polText} - x:${comp.x}, y:${comp.y}, w:${comp.width}, h:${comp.height}`;
  }

  productForm = document.getElementById('productForm')
  productForm.addEventListener('submit', async (e) => {
    e.preventDefault();
  
    // if (!annotations.goldenImage) {
    //   alert('Selecione uma imagem Golden antes de salvar.');
    //   return;
    // }
  
    const formData = new FormData(productForm);
  
    formData.set('fiducials', JSON.stringify(annotations.fiducials));
    formData.set('components', JSON.stringify(annotations.components));
  
    // NOVO: manda os templates de pacote (ROI + rects do corpo)
    formData.set('package_templates', JSON.stringify(newPackageTemplates || {}));
  
    try {
      const resp = await fetch('/add_product', {
        method: 'POST',
        body: formData,
      });
  
      const data = await resp.json();
      if (!resp.ok) {
        console.error('Erro ao salvar produto:', data);
        alert(data.error || 'Erro ao salvar produto.');
        return;
      }
  
      alert('Produto salvo com sucesso!');
      window.location.href = `/inspect?product_id=${data.product_id}`;
    } catch (err) {
      console.error('Erro de rede ao salvar produto:', err);
      alert('Erro de rede ao salvar produto.');
    }
  });
});
