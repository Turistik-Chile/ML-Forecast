# Interfaz basada en shadcn/ui + Framer Motion

Este tutorial resume los pasos aplicados para migrar la aplicación a la pila de componentes de **shadcn/ui** usando **Tailwind CSS**, añadir microinteracciones con **Framer Motion**, incorporar theming claro/oscuro y mover el gráfico a la integración recomendada (Recharts).

## 1. Instalación básica

1. Instala Tailwind y su toolchain:
   ```bash
   npm install -D tailwindcss@3 postcss autoprefixer
   ```
2. Añade las dependencias de shadcn/ui y utilidades:
   ```bash
   npm install class-variance-authority tailwind-merge clsx lucide-react \
     @radix-ui/react-slot @radix-ui/react-select @radix-ui/react-scroll-area \
     tailwindcss-animate
   ```
3. Añade Recharts, Framer Motion y React Icons:
   ```bash
   npm install recharts framer-motion react-icons
   ```
4. Crea/actualiza `tailwind.config.js` y `postcss.config.js` (ambos ya se encuentran en `frontend/`). La configuración replica la propuesta oficial de shadcn/ui y habilita `darkMode: ["class"]`.
5. Crea `src/index.css` con las directivas `@tailwind base`, `components`, `utilities` y define estilos base para `:root`, `.dark` y `body`. Importa el archivo en `src/main.tsx`.

## 2. Componentes compartidos

En `src/components/ui/` se añadieron los “primitivos” reutilizables:

| Componente | Uso principal |
| ---------- | ------------- |
| `button.tsx` | Botones con variantes (`default`, `secondary`, `ghost`, etc.) basados en `class-variance-authority`. |
| `card.tsx` | Contenedores elevables para formularios, métricas y tablas. |
| `input.tsx`, `label.tsx` | Campos de formulario con estilos consistentes. |
| `select.tsx` | Selectores Radix UI estilizados con Tailwind. |
| `badge.tsx` | Etiquetas para lags y categorías activas. |
| `scroll-area.tsx` | Contenedor con scrollbar personalizado para la lista de lags. |
| `alert.tsx` | Mensajes de estado y advertencias globales. |

Todos utilizan el helper `cn` (`src/lib/utils.ts`), que combina `clsx` con `tailwind-merge`.

## 3. Formularios y layout

* `App.tsx` reemplazó la antigua hoja CSS por utilidades Tailwind. Cada bloque de resultados aparece con transiciones suaves (`motion.section`). Se añadió un selector de tema (claro/oscuro) que:
  * Usa la clase `dark` de Tailwind.
  * Persiste en `localStorage` y respeta `prefers-color-scheme`.
  * Emplea **react-icons** (`FiSun`/`FiMoon`) para el botón flotante.
* `ForecastForm.tsx` se reconstruyó como un `Card` único con grid responsivo:
  * Selectores shadcn para pipeline, ubicación y modelo.
  * Inputs `date` cuando se usa turismo.
  * Selector de categorías basado en `Badge` + `ScrollArea`.
  * Gestor de lags con tokens editables, validaciones contextuales y alertas cuando el historial no cubre los lags (en línea con el padding máximo de 30 días en el backend).
* `StatusMessage`, `MetricsSection` y `ObservationsTable` adoptan `Card`, `Badge` y `Alert` para unificar la estética.

## 4. Gráfico con Recharts

shadcn/ui recomienda usar librerías externas para visualizaciones, así que sustituí Chart.js por **Recharts** (`AreaChart` + `ResponsiveContainer`). El nuevo `ForecastChart.tsx` incorpora:

* Doble área (real/predicción) con degradados alineados a la paleta.
* Tooltips y leyenda nativos.
* Contenedor `Card` para integrarlo visualmente al dashboard.

Chart.js dejó de ser una dependencia del proyecto.

## 5. Animaciones y theming

* `PageHeader`, `ForecastForm` y los resultados usan Framer Motion para introducir suavemente cada sección.
* El theming se controla en `App.tsx`: al cambiar de modo se actualiza la clase `dark`, se guarda en `localStorage` y se escucha el `matchMedia` para seguir la preferencia del sistema.
* React Icons se utiliza para el botón de toggle y puede ampliarse para otros íconos si es necesario.

## 6. Scripts útiles

Desde `frontend/`:

```bash
npm install    # asegura dependencias actualizadas
npm run dev    # entorno de desarrollo con Vite
npm run build  # compila para producción (verificado tras la migración)
```

## 7. Checklist aplicado

- [x] Configuración de Tailwind + PostCSS y utilidades internas.
- [x] Reemplazo de los elementos visuales por componentes shadcn/ui.
- [x] Gestor de lags renovado con ScrollArea, badges y validaciones.
- [x] Integración de Recharts como gráfico principal.
- [x] Animaciones clave con Framer Motion y selector de tema claro/oscuro persistente.
- [x] Uso de `react-icons` para el toggle y futuras necesidades de iconografía.
- [x] Documentación de todo el flujo (este archivo).

Con esto la interfaz queda lista para seguir incorporando componentes del catálogo shadcn/ui (`dialog`, `tabs`, etc.) según lo requiera el producto. ***
