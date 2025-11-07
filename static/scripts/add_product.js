    // --- Variáveis de Elementos DOM (Atualizado) ---
    const imagePreview = document.getElementById('imagePreview');
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const fiducialModal = document.getElementById('fiducialModal');
    const componentModal = document.getElementById('componentModal');
    const compPackageSelect = document.getElementById('compPackage');
    const newCompPackageInput = document.getElementById('newCompPackage');
    const compPresenceInput = document.getElementById('compPresenceThreshold');
    const compSsimInput = document.getElementById('compSsimThreshold');
    
    // Modal de Ajuste (Definição)
    const bodyAdjustModal = document.getElementById('bodyAdjustModal');
    const bodyDefinitionContainer = document.getElementById('bodyDefinitionContainer');
    const bodyDefinitionStatus = document.getElementById('bodyDefinitionStatus');
    const defineBodyBtn = document.getElementById('defineBodyBtn');
    const bodyDefinitionDone = document.getElementById('bodyDefinitionDone');
    const confirmBodyBtn = document.getElementById('confirmBodyBtn');
    const cancelBodyBtn = document.getElementById('cancelBodyBtn');
    const bodyCanvasContainer = document.getElementById('bodyCanvasContainer');
    const bodyAdjustLoading = document.getElementById('bodyAdjustLoading');

    // Novo Modal de Confirmação
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


    // --- Estado da Aplicação (Atualizado) ---
    let currentMode = null;
    let isDrawing = false;
    let startX, startY;
    let annotations = { fiducials: [], components: [] };
    let lastAnnotationType = null; 
    let tempAnnotation = null; 
    let currentComponentRect = null; 
    let currentComponentROI_b64 = null; 
    let finalBodyRect = null; 
    
    // ATUALIZADO: Objeto para rastrear templates definidos nesta sessão
    let newPackageTemplates = {};

    // Estado do Canvas de Ajuste (Sem Alterações)
    let bodyCanvas, bodyCtx, bodyImg;
    let adjustableRect = { x: 10, y: 10, width: 50, height: 50 };
    let isResizing = false;
    let isDragging = false;
    let resizeHandle = null; 
    const handleSize = 8;
    let bodyScaleX = 1.0;
    let bodyScaleY = 1.0;

    // Estado do Canvas de Confirmação
    let confirmCanvas, confirmCtx, confirmImg;
    let confirmScaleX = 1.0;
    let confirmScaleY = 1.0;

    // --- Funções de UI (previewImage, drawAnnotations, getMousePos, eventos canvas) (Sem Alterações) ---
    function previewImage(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
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
    }

    function drawAnnotations() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scaleX = canvas.width / imagePreview.naturalWidth;
        const scaleY = canvas.height / imagePreview.naturalHeight;

        if (tempAnnotation && tempAnnotation.type === 'fiducial_box') {
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.strokeRect(tempAnnotation.x * scaleX, tempAnnotation.y * scaleY, tempAnnotation.width * scaleX, tempAnnotation.height * scaleY);
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
            if (comp.final_body_rect) { // (Só para comps que definem)
                ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'; 
                const rect = comp.final_body_rect;
                const regionX = (comp.x + rect.x) * scaleX;
                const regionY = (comp.y + rect.y) * scaleY;
                ctx.fillRect(regionX, regionY, rect.width * scaleX, rect.height * scaleY);
            }
        });
    }
    
    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    canvas.addEventListener('mousedown', function(e) {
        if (!currentMode) {
            alert('Selecione um modo (Fiducial ou Componente) primeiro.');
            return;
        }
        const pos = getMousePos(e);
        isDrawing = true;
        startX = pos.x;
        startY = pos.y;
    });

    canvas.addEventListener('mousemove', function(e) { /* (vazio) */ });
    
    canvas.addEventListener('mouseup', async function(e) {
        if (!isDrawing) return;
        isDrawing = false;
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
            showComponentModal(naturalRect);
        }
    });

    // --- Lógica dos Modais ---
    
    // showFiducialModal (Sem Alterações)
    async function showFiducialModal(rect) {
        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = rect.width;
        croppedCanvas.height = rect.height;
        const croppedCtx = croppedCanvas.getContext('2d');
        croppedCtx.drawImage(imagePreview, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
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
                const closeModal = (isConfirmed = false) => {
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
    
    // compPackageSelect 'change' (ATUALIZADO)
    compPackageSelect.addEventListener('change', () => {
        const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex];
        
        // Esconde ambos os containers por padrão
        bodyDefinitionContainer.classList.add('hidden');
        bodyConfirmationContainer.classList.add('hidden');

        if (compPackageSelect.value === '--new--') {
            newCompPackageInput.classList.remove('hidden');
            compPresenceInput.value = 0.35; 
            compSsimInput.value = 0.60;
            
            // Fluxo de DEFINIÇÃO
            bodyDefinitionContainer.classList.remove('hidden');
            bodyDefinitionStatus.textContent = "Pacote novo. O corpo DEVE ser definido.";
            bodyDefinitionDone.classList.add('hidden');
            defineBodyBtn.classList.remove('hidden');
            finalBodyRect = null; 
        } else {
            newCompPackageInput.classList.add('hidden');
            
            // Preenche limiares
            if (selectedOption.dataset.presence) {
                compPresenceInput.value = parseFloat(selectedOption.dataset.presence).toFixed(2);
                compSsimInput.value = parseFloat(selectedOption.dataset.ssim).toFixed(2);
            }
            
            // Verifica se o corpo e o TAMANHO DA ROI já foram definidos
            const isBodyDefined = selectedOption.dataset.bodyDefined === 'true';
            // Checa se o tamanho da ROI já foi salvo (pelo primeiro componente)
            const isRoiSizeDefined = parseInt(selectedOption.dataset.templateRoiWidth || '0') > 0;

            // ATUALIZAÇÃO: Checa também o newPackageTemplates, caso tenha sido definido nesta sessão
            const isDefinedInSession = !!newPackageTemplates[selectedOption.value];

            if ((isBodyDefined && isRoiSizeDefined) || isDefinedInSession) {
                // Fluxo de CONFIRMAÇÃO
                bodyConfirmationContainer.classList.remove('hidden');
                bodyConfirmationDone.classList.add('hidden');
                confirmBodyBtn_Flow.classList.remove('hidden');
                finalBodyRect = null;
            } else {
                // Fluxo de DEFINIÇÃO (mesmo que o pacote exista, falta info)
                bodyDefinitionContainer.classList.remove('hidden');
                bodyDefinitionStatus.textContent = "Este pacote não tem corpo ou tamanho de ROI. O corpo DEVE ser definido.";
                bodyDefinitionDone.classList.add('hidden');
                defineBodyBtn.classList.remove('hidden');
                finalBodyRect = null;
            }
        }
    });

    // --- (NOVO) Helper para salvar dados no dropdown ---
    function updatePackageDropdown(packageName, width, height) {
        let optionExists = false;
        for (let i = 0; i < compPackageSelect.options.length; i++) {
            if (compPackageSelect.options[i].value === packageName) {
                optionExists = true;
                const opt = compPackageSelect.options[i];
                opt.dataset.bodyDefined = 'true';
                opt.dataset.templateRoiWidth = width;
                opt.dataset.templateRoiHeight = height;
                console.log(`Pacote '${packageName}' atualizado com w:${width}, h:${height}`);
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
            
            const newOptionEntry = compPackageSelect.querySelector('option[value="--new--"]');
            compPackageSelect.insertBefore(newOption, newOptionEntry);
            console.log(`Pacote '${packageName}' adicionado com w:${width}, h:${height}`);
        }
    }

    // --- (NOVO) Helper para salvar a anotação ---
    function pushComponentAnnotation(packageName, isDefining) {
        const newComponent = {
            name: document.getElementById('compName').value.trim() || `COMP${annotations.components.length + 1}`,
            package: packageName,
            rotation: parseInt(document.getElementById('compRotation').value),
            // currentComponentRect já foi atualizado (seja bruto ou corrigido)
            x: currentComponentRect.x, 
            y: currentComponentRect.y, 
            width: currentComponentRect.width, 
            height: currentComponentRect.height,
        };

        // Anexa os dados do corpo SE ele foi definido nesta sessão
        if (isDefining) {
            newComponent.final_body_rect = finalBodyRect;
            newComponent.component_roi_b64 = currentComponentROI_b64;
            
            // Salva o tamanho da ROI deste componente no dropdown
            updatePackageDropdown(packageName, currentComponentRect.width, currentComponentRect.height);
            
            // ATUALIZADO: Salva o template no objeto JS
            newPackageTemplates[packageName] = {
                roi_b64: currentComponentROI_b64,
                body_rect: finalBodyRect,
                roi_width: currentComponentRect.width,
                roi_height: currentComponentRect.height
            };
        }

        annotations.components.push(newComponent);
        lastAnnotationType = { type: 'component' };
    }


    // showComponentModal (ATUALIZADO)
    function showComponentModal(rect) {
        // Reseta o estado
        document.getElementById('compName').value = `COMP${annotations.components.length + 1}`;
        compPackageSelect.value = "";
        newCompPackageInput.value = "";
        newCompPackageInput.classList.add('hidden');
        document.getElementById('compRotation').value = "0";
        compPresenceInput.value = "";
        compSsimInput.value = "";
        bodyDefinitionContainer.classList.add('hidden');
        bodyConfirmationContainer.classList.add('hidden');

        // Armazena a ROI bruta
        currentComponentRect = rect;
        finalBodyRect = null; 
        
        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = rect.width;
        croppedCanvas.height = rect.height;
        const croppedCtx = croppedCanvas.getContext('2d');
        croppedCtx.drawImage(imagePreview, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
        currentComponentROI_b64 = croppedCanvas.toDataURL('image/png');
        
        componentModal.classList.remove('hidden');

        // Botão Salvar agora é um dispatcher
        const handleSave = () => {
            let packageName = compPackageSelect.value;
            if (packageName === '--new--') {
                packageName = newCompPackageInput.value.trim();
                if (!packageName) {
                    alert("Por favor, insira um nome para o novo pacote.");
                    return;
                }
            }
            if (!packageName) {
                alert("Selecione ou cadastre um pacote.");
                return;
            }

            const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex];
            const isBodyDefined = selectedOption.dataset.bodyDefined === 'true';
            const isRoiSizeDefined = parseInt(selectedOption.dataset.templateRoiWidth || '0') > 0;
            const isDefinedInSession = !!newPackageTemplates[packageName];

            // Cenário 1: É um pacote que precisa de DEFINIÇÃO
            if ((!isBodyDefined || !isRoiSizeDefined) && !isDefinedInSession) {
                if (!finalBodyRect) {
                    alert("Este pacote é novo ou não tem um corpo/tamanho definidos. Por favor, clique em 'Sugerir & Ajustar Corpo' e confirme.");
                    return;
                }
                // Se chegou aqui, finalBodyRect está definido. Salva e fecha.
                pushComponentAnnotation(packageName, true); // true = está definindo
                closeModal();
            } 
            // Cenário 2: É um pacote que precisa de CONFIRMAÇÃO
            else {
                if (bodyConfirmationDone.classList.contains('hidden')) {
                    alert("Este pacote já existe. Por favor, clique em 'Encontrar & Confirmar Corpo' para ajustar a ROI.");
                    return;
                }
                // Se chegou aqui, a confirmação foi feita. Salva e fecha.
                // a ROI já foi corrigida pelo fluxo de confirmação.
                pushComponentAnnotation(packageName, false); // false = não está definindo
                closeModal();
            }
        };

        const closeModal = () => {
            componentModal.classList.add('hidden');
            document.getElementById('saveComponentBtn').onclick = null;
            document.getElementById('cancelComponentBtn').onclick = null;
            currentComponentRect = null;
            currentComponentROI_b64 = null;
            finalBodyRect = null;
            drawAnnotations();
            updateAnnotationsList();
        };

        document.getElementById('saveComponentBtn').onclick = handleSave;
        document.getElementById('cancelComponentBtn').onclick = closeModal;
    }


    // --- Lógica do Fluxo de DEFINIÇÃO (bodyAdjustModal) (Sem Alterações) ---
    defineBodyBtn.addEventListener('click', async () => {
        if (!currentComponentROI_b64) {
            alert("Erro: ROI do componente não encontrada.");
            return;
        }
        defineBodyBtn.disabled = true;
        defineBodyBtn.textContent = "Sugerindo...";
        try {
            const response = await fetch('/suggest_body', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ component_roi_b64: currentComponentROI_b64 })
            });
            const suggestion = await response.json();
            if (!response.ok) throw new Error(suggestion.error);
            openBodyAdjustModal(currentComponentROI_b64, suggestion);
        } catch (error) {
            alert("Erro ao sugerir corpo: " + error.message);
        } finally {
            defineBodyBtn.disabled = false;
            defineBodyBtn.textContent = "1. Sugerir & Ajustar Corpo";
        }
    });
    
    function openBodyAdjustModal(roi_b64, suggestedRect) {
        bodyAdjustLoading.style.display = 'block';
        bodyCanvasContainer.innerHTML = ''; 
        bodyAdjustModal.classList.remove('hidden');

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

            adjustableRect = suggestedRect;
            
            bodyCanvas.addEventListener('mousedown', onBodyCanvasMouseDown);
            bodyCanvas.addEventListener('mousemove', onBodyCanvasMouseMove);
            bodyCanvas.addEventListener('mouseup', onBodyCanvasMouseUp);
            bodyCanvas.addEventListener('mouseout', onBodyCanvasMouseUp);
            
            drawBodyCanvas();
        }
    }

    function drawBodyCanvas() {
        if (!bodyCanvas) return;
        bodyCtx.clearRect(0, 0, bodyCanvas.width, bodyCanvas.height);
        bodyCtx.drawImage(bodyImg, 0, 0, bodyCanvas.width, bodyCanvas.height);

        const rect = adjustableRect;
        bodyCtx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
        bodyCtx.lineWidth = 2;
        bodyCtx.strokeRect(rect.x * bodyScaleX, rect.y * bodyScaleY, rect.width * bodyScaleX, rect.height * bodyScaleY);

        bodyCtx.fillStyle = 'rgba(0, 255, 0, 0.9)';
        const handles = getResizeHandles(rect, bodyScaleX, bodyScaleY);
        for (const handle in handles) {
            const h = handles[handle];
            bodyCtx.fillRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
        }
    }

    function getResizeHandles(rect, scaleX, scaleY) {
        return {
            tl: { x: rect.x * scaleX, y: rect.y * scaleY },
            tr: { x: (rect.x + rect.width) * scaleX, y: rect.y * scaleY },
            bl: { x: rect.x * scaleX, y: (rect.y + rect.height) * scaleY },
            br: { x: (rect.x + rect.width) * scaleX, y: (rect.y + rect.height) * scaleY }
        };
    }

    function getMousePosOnBodyCanvas(e) {
        const rect = bodyCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    function onBodyCanvasMouseDown(e) {
        const pos = getMousePosOnBodyCanvas(e);
        if (bodyScaleX === 1.0 && bodyScaleY === 1.0) console.warn("Escalas do canvas não definidas!");

        const handles = getResizeHandles(adjustableRect, bodyScaleX, bodyScaleY);
        resizeHandle = null;
        for (const handle in handles) {
            const h = handles[handle];
            if (Math.abs(pos.x - h.x) < handleSize && Math.abs(pos.y - h.y) < handleSize) {
                resizeHandle = handle;
                isResizing = true;
                startX = pos.x;
                startY = pos.y;
                return;
            }
        }
        
        const rect = adjustableRect;
        if (pos.x > rect.x * bodyScaleX && pos.x < (rect.x + rect.width) * bodyScaleX &&
            pos.y > rect.y * bodyScaleY && pos.y < (rect.y + rect.height) * bodyScaleY) {
            isDragging = true;
            startX = pos.x;
            startY = pos.y;
            return;
        }
    }

    function onBodyCanvasMouseMove(e) {
        const pos = getMousePosOnBodyCanvas(e);
        const dx_img = (pos.x - startX) / bodyScaleX;
        const dy_img = (pos.y - startY) / bodyScaleY;

        if (isResizing) {
            const rect = adjustableRect;
            if (resizeHandle.includes('t')) { rect.y += dy_img; rect.height -= dy_img; }
            if (resizeHandle.includes('l')) { rect.x += dx_img; rect.width -= dx_img; }
            if (resizeHandle.includes('b')) { rect.height += dy_img; }
            if (resizeHandle.includes('r')) { rect.width += dx_img; }
            startX = pos.x;
            startY = pos.y;
        } else if (isDragging) {
            adjustableRect.x += dx_img;
            adjustableRect.y += dy_img;
            startX = pos.x;
            startY = pos.y;
        }
        
        if(isResizing || isDragging) {
            drawBodyCanvas();
        }
    }
    
    function onBodyCanvasMouseUp(e) {
        if (isResizing && (adjustableRect.width < 0 || adjustableRect.height < 0)) {
            if(adjustableRect.width < 0) {
                adjustableRect.x += adjustableRect.width;
                adjustableRect.width *= -1;
            }
            if(adjustableRect.height < 0) {
                adjustableRect.y += adjustableRect.height;
                adjustableRect.height *= -1;
            }
        }
        isResizing = false;
        isDragging = false;
        resizeHandle = null;
        drawBodyCanvas();
    }

    confirmBodyBtn.addEventListener('click', () => {
        finalBodyRect = {
            x: Math.round(adjustableRect.x),
            y: Math.round(adjustableRect.y),
            width: Math.round(adjustableRect.width),
            height: Math.round(adjustableRect.height)
        };
        bodyAdjustModal.classList.add('hidden');
        bodyCanvasContainer.innerHTML = '';
        bodyCanvasContainer.style.height = 'auto'; 
        bodyCanvas = null;
        bodyDefinitionDone.classList.remove('hidden');
        defineBodyBtn.classList.add('hidden');
    });

    cancelBodyBtn.addEventListener('click', () => {
        bodyAdjustModal.classList.add('hidden');
        bodyCanvasContainer.innerHTML = '';
        bodyCanvasContainer.style.height = 'auto'; 
        bodyCanvas = null;
    });


    // --- Lógica do NOVO Fluxo de CONFIRMAÇÃO (bodyConfirmModal) ---
    
    // 1. Botão "Encontrar & Confirmar Corpo" (ATUALIZADO)
    confirmBodyBtn_Flow.addEventListener('click', async () => {
        const selectedOption = compPackageSelect.options[compPackageSelect.selectedIndex];
        const packageName = selectedOption.value;
        
        if (!packageName) {
            alert("Erro: Pacote não selecionado.");
            return;
        }

        confirmBodyBtn_Flow.disabled = true;
        confirmBodyBtn_Flow.textContent = "Procurando...";
        bodyConfirmLoading.style.display = 'block';
        bodyConfirmCanvasContainer.innerHTML = '';
        bodyConfirmTemplateImg.src = '';

        let template_roi_size;
        let url = '';
        let payload = {};

        // Verifica se o template foi definido NESTA sessão
        if (newPackageTemplates[packageName]) {
            console.log("Usando template do cache JS");
            const templateData = newPackageTemplates[packageName];
            url = '/find_body_in_roi_with_template'; // Rota 1 (JS)
            payload = {
                component_roi_b64: currentComponentROI_b64, // ROI C2
                template_roi_b64: templateData.roi_b64,     // ROI C1
                template_body_rect: templateData.body_rect  // Body C1
            };
            template_roi_size = {
                width: templateData.roi_width,
                height: templateData.roi_height
            };
        } else {
            // Template deve estar no DB
            console.log("Usando template do DB");
            url = '/find_body_in_roi'; // Rota 2 (DB)
            payload = {
                component_roi_b64: currentComponentROI_b64, 
                package_name: packageName 
            };
            template_roi_size = {
                width: parseInt(selectedOption.dataset.templateRoiWidth),
                height: parseInt(selectedOption.dataset.templateRoiHeight)
            };
        }
        
        if (template_roi_size.width === 0) {
             alert("Erro: Tamanho da ROI para este pacote é desconhecido. Defina o pacote primeiro.");
             confirmBodyBtn_Flow.disabled = false;
             confirmBodyBtn_Flow.textContent = "1. Encontrar & Confirmar Corpo";
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
            
            // Abre o modal de confirmação com os dados
            openBodyConfirmModal(result.body_rect, result.template_b64, template_roi_size);

        } catch (error) {
            alert("Erro ao encontrar corpo: " + error.message);
        } finally {
            confirmBodyBtn_Flow.disabled = false;
            confirmBodyBtn_Flow.textContent = "1. Encontrar & Confirmar Corpo";
        }
    });

    // 2. Abrir e inicializar o modal de confirmação (Sem Alterações)
    function openBodyConfirmModal(found_body_rect, template_b64, template_roi_size) {
        bodyConfirmModal.classList.remove('hidden');
        bodyConfirmLoading.style.display = 'block';
        bodyConfirmCanvasContainer.innerHTML = '';
        bodyConfirmTemplateImg.src = 'data:image/png;base64,' + template_b64;

        confirmImg = new Image();
        confirmImg.src = currentComponentROI_b64; // A ROI bruta
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

            // Desenha a imagem
            confirmCtx.drawImage(confirmImg, 0, 0, canvasWidth, canvasHeight);
            
            // Desenha a caixa verde (corpo encontrado)
            confirmCtx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
            confirmCtx.lineWidth = 2;
            confirmCtx.strokeRect(
                found_body_rect.x * confirmScaleX,
                found_body_rect.y * confirmScaleY,
                found_body_rect.width * confirmScaleX,
                found_body_rect.height * confirmScaleY
            );
            
            bodyConfirmLoading.style.display = 'none';
        }

        // 3. Lógica dos botões de confirmação/cancelamento
        confirmBodyPositionBtn.onclick = () => {
            // Calcula o centro do corpo (em pixels da ROI bruta)
            const body_center_x = found_body_rect.x + found_body_rect.width / 2;
            const body_center_y = found_body_rect.y + found_body_rect.height / 2;
            
            // Calcula o centro absoluto (em pixels da imagem golden)
            const abs_center_x = currentComponentRect.x + body_center_x;
            const abs_center_y = currentComponentRect.y + body_center_y;
            
            // Calcula o novo (x, y) da ROI (caixa azul) usando o tamanho padrão
            const final_roi_x = abs_center_x - (template_roi_size.width / 2);
            const final_roi_y = abs_center_y - (template_roi_size.height / 2);
            
            // ATUALIZA A ROI ATUAL (caixa azul)
            currentComponentRect = {
                x: Math.round(final_roi_x),
                y: Math.round(final_roi_y),
                width: template_roi_size.width,
                height: template_roi_size.height
            };
            
            // Limpa os dados de definição (pois não estamos definindo)
            currentComponentROI_b64 = null;
            finalBodyRect = null;
            
            // Fecha o modal e atualiza a UI do modal de componente
            bodyConfirmModal.classList.add('hidden');
            bodyConfirmationDone.classList.remove('hidden');
            confirmBodyBtn_Flow.classList.add('hidden');
        };

        cancelBodyPositionBtn.onclick = () => {
            bodyConfirmModal.classList.add('hidden');
            // Não faz nada, usuário pode tentar de novo ou cancelar
        };
    }


    // --- Funções de Controle de Modo e UI (Finais) (Sem Alterações) ---
    function setActiveMode(mode, instructionsText) {
        currentMode = mode;
        ['fiducialModeBtn', 'componentModeBtn'].forEach(id => {
            document.getElementById(id).classList.remove('bg-indigo-600', 'text-white');
            document.getElementById(id).classList.add('btn-secondary');
        });
        const btn = document.getElementById(`${mode}ModeBtn`);
        btn.classList.remove('btn-secondary');
        btn.classList.add('bg-indigo-600', 'text-white');
        document.getElementById('instructions').textContent = instructionsText;
    }

    document.getElementById('fiducialModeBtn').addEventListener('click', () => setActiveMode('fiducial', "Arraste na imagem para marcar a área de um fiducial."));
    document.getElementById('componentModeBtn').addEventListener('click', () => setActiveMode('component', "Arraste para marcar a área de um componente (caixa azul)."));
    
    document.getElementById('undoBtn').addEventListener('click', () => {
        if (!lastAnnotationType) return;
        if (lastAnnotationType.type === 'fiducial') {
            annotations.fiducials.pop();
        } else if (lastAnnotationType.type === 'component') {
            annotations.components.pop();
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
            fiducialsList.innerHTML += `<li class="text-sm">Fiducial ${i+1}: (x:${f.x}, y:${f.y}, r:${f.r})</li>`;
        });
        annotations.components.forEach((c, i) => {
            let bodyInfo = (c.final_body_rect) ? 'com corpo definido' : 'usando corpo do pacote';
            if (!c.package) bodyInfo = "sem pacote";
            componentsList.innerHTML += `<li class="text-sm">${c.name} (${c.package}, ${c.rotation}°): ${bodyInfo}</li>`;
        });
    }
    
    // Lógica de Submit (Sem Alterações)
    document.getElementById('productForm').addEventListener('submit', async function(event) {
        event.preventDefault();
        if (annotations.fiducials.length < 2) {
            alert("Você precisa marcar no mínimo 2 fiduciais para um alinhamento preciso.");
            return;
        }
        const packagesComCorpoDefinido = new Set(Object.keys(newPackageTemplates));
        let erroPacote = null;
        
        document.querySelectorAll('#compPackage option').forEach(opt => {
            if(opt.value && opt.value !== '--new--' && opt.dataset.bodyDefined === 'true') {
                packagesComCorpoDefinido.add(opt.value);
            }
        });
        for (const comp of annotations.components) {
            const packageName = comp.package;
            if (!packagesComCorpoDefinido.has(packageName)) {
                // Este check agora é mais robusto
                if (!comp.final_body_rect || !comp.component_roi_b64) {
                    erroPacote = `O componente '${comp.name}' é o primeiro do novo pacote '${packageName}'. Você *deve* definir o corpo para ele no modal.`;
                    break; 
                } else {
                    // Adiciona ao set caso tenha sido definido mas o loop principal perdeu
                    packagesComCorpoDefinido.add(packageName); 
                }
            }
        }
        if (erroPacote) {
            alert(erroPacote);
            return; 
        }
        const formData = new FormData(event.target);
        formData.set('fiducials', JSON.stringify(annotations.fiducials));
        formData.set('components', JSON.stringify(annotations.components));
        const button = document.getElementById('saveButton');
        button.disabled = true;
        document.getElementById('buttonText').classList.add('hidden');
        document.getElementById('spinner').classList.remove('hidden');
        try {
            const response = await fetch('/add_product', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                alert('Produto cadastrado com sucesso!');
                window.location.href = '/';
            } else {
                throw new Error(result.error || 'Erro desconhecido.');
            }
        } catch (error) {
            alert('Erro ao cadastrar produto: ' + error.message);
        } finally {
            button.disabled = false;
            document.getElementById('buttonText').classList.remove('hidden');
            document.getElementById('spinner').classList.add('hidden');
        }
    });