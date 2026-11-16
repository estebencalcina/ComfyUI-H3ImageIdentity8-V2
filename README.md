# ComfyUI-H3ImageIdentity8-V2

**Nueva extensión independiente:** nombres de nodos distintos a `ComfyUI-H3ImageNativeRefMod`. No modifica ni desinstala el nodo antiguo. Instálala en otra carpeta dentro de `custom_nodes`.

## Compatibilidad con identidades anteriores

**IMPORTANTE:** utiliza la misma carpeta persistente que el creador anterior:
`ComfyUI/models/h3_image_native_refmods/`.

Lee archivos `.safetensors` con metadata `format = h3-image-native-multiref-v2` creados con el nodo anterior de 8 imágenes. **No convierte archivos de RefMod de H3 Video** ni el formato antiguo de una sola imagen. No sobrescribe ni mueve identidades existentes. `overwrite_existing` está DESACTIVADO por defecto en el creador nuevo.

## Qué incluye

- `H3 IMAGE IDENTITY V2 · CREATE 8 PHOTOS`: crea una identidad con 1 a 8 fotos, usando el VAE nativo H3-Image.
- `H3 IMAGE IDENTITY V2 · LOAD / REFRESH`: lee las identidades ya guardadas. Botón JS **Actualizar identidades** y campo `manual_filename` por si no aparece en el selector.
- `H3 IMAGE IDENTITY V2 · T2I CONDITIONING`: integra imágenes y latentes de referencia en `minimax_refs`.

## Instalación

1. Crea el nuevo repositorio GitHub `ComfyUI-H3ImageIdentity8-V2` y sube los archivos, no el ZIP sin extraer.
2. En Modal/ComfyUI, instala como carpeta nueva: `ComfyUI/custom_nodes/ComfyUI-H3ImageIdentity8-V2/` (o clona el repositorio).
3. Reinicia el proceso de ComfyUI para registrar el backend y su nueva ruta API. Recarga la pestaña del navegador con Ctrl+F5.
4. Importa `workflows/02_V2_T2I_CARGAR_IDENTIDAD.json`.
5. En el nodo de carga, pulsa `↻ Actualizar identidades`. El campo `manual_filename` comienza con `Personaje_01pks.safetensors`, el nombre visto en tu captura, para que puedas cargar ese archivo aunque el desplegable no se actualice. **Cuando uses el desplegable para otra identidad, vacía `manual_filename`; si no, siempre tendrá preferencia el nombre manual.**
6. Importa `workflows/01_V2_CREAR_IDENTIDAD_8_FOTOS.json` solo si necesitas CREAR una identidad nueva. No hace falta recrear la que ya tienes.

## Protección de datos

- `Create` guarda en la carpeta existente y se niega a sobrescribir si existe un nombre igual, salvo que actives explícitamente `overwrite_existing`.
- La nueva extensión no cambia ni elimina tus modelos ni tu Volume Modal.
- Para que el nuevo cargador pueda ver la identidad anterior, **el volumen persistente debe montarse en la misma ubicación** donde `folder_paths.models_dir` la ve.
- No publiques archivos de identidad privados en GitHub; solo código y JSON.

## Estado experimental

El diseño de integración nativa aún necesita una prueba de generación real con GPU; que el loader funcione no garantiza fidelidad facial. El sistema guarda referencias visuales y latentes completos, **no** un RefMod comprimido entrenado.


## Actualización 2026-11: mayor influencia facial

El identificador del nodo **sigue siendo `H3Img8V2Condition`**: no se instalan nodos duplicados, basta con actualizar este mismo repositorio GitHub y reiniciar ComfyUI. El formato `h3-image-native-multiref-v2` y la ubicación de `Personaje_01pks.safetensors` **no cambian**. El cargador lee el archivo existente; ningún proceso sobrescribe identidades durante la generación.

Controles nuevos de `H3Img8V2Condition`:

- `reference_strength` (0–1): retención espacial como el RefMod original. **1.0 es la máxima retención**, no existe un coeficiente nativo de 1.5x; valores menores difuminan el latente. Un valor de 0 omite referencias Qwen y DiT.
- `reference_copies` (1–3): repite la secuencia de bloques en el DiT. 2 copies de ocho fotos = 16 bloques; 3 = 24. **Es una heurística de peso por repetición, no una garantía de identidad.** Aumenta VRAM y tiempo.
- `face_priority_view` (0–8): selecciona una de las vistas almacenadas; 0 desactiva prioridad. Elige la foto con rostro frontal más nítido.
- `face_priority_copies` (0–3): añade copias de esa foto al final de los bloques.
- `auto_identity_hint`: antepone instrucciones explícitas sobre que `<Picture 1>` ... `<Picture N>` representan a la misma persona y etiqueta imágenes con el mismo formato del nodo oficial de H3-Image.

**Prueba recomendada:** usar `02_V2_T2I_CARGAR_IDENTIDAD.json` con 2 copias de las 8 vistas y 1 copia extra de la primera foto (17 bloques), retención 1.0, seed fijo. Para comparación directa, `03_V2_COMPARAR_BASELINE_1_COPIA.json` usa 8 bloques con igual prompt, resolución y seed. Si hay OOM, poner `reference_copies=1` y `face_priority_copies=0`.

**Calidad de la identidad existente:** el archivo fue creado con referencias de lado corto 768. Cambiar el tamaño en el creador a 1024 afecta únicamente identidades **nuevas** y no recupera detalle de las ya guardadas. Nunca sobrescribir la identidad original sin copia de seguridad.

## Validación

Sintaxis Python y JSON verificados localmente. La identidad y el comportamiento de atención requieren pruebas en GPU/ComfyUI, y no se garantiza 100% de similitud.
