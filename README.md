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
