animation = """
<style>
:root{
  --td11-start: 0.1;
  --td11-end: 0.5;
  --td12-start: 0.0;
  --td12-end: 0.2;
  --td13-start: 0.2;
  --td13-end: 0.4;
  --td21-start: 0.5;
  --td21-end: 0.8;
  --td22-start: 0.2;
  --td22-end: 0.5;
  --td23-start: 0.4;
  --td23-end: 0.1;
  --td31-start: 0.8;
  --td31-end: 0.4;
  --td32-start: 0.5;
  --td32-end: 0.3;
  --td33-start: 0.1;
  --td33-end: 0.2;
}
body {
  display: grid;
  place-items: center;
  height: 100vh;
  margin: 0;
  background-color: #222;
}
.area-chart {
  /* Reset */
  margin: 0;
  padding: 0;
  border: 0;
  margin-left: auto;
  margin-right: auto;
  margin-bottom: 80px;

  /* Dimensions */
  width: 100%;
  max-width: var(--chart-width, 600px);
  height: var(--chart-height, 300px);
}
.area-chart tbody {
  width: 100%;
  height: var(--chart-height, 300px);

  /* Layout */
  display: flex;
  justify-content: stretch;
  align-items: stretch;
  flex-direction: row;
}
.area-chart tr {
  position: relative;
  border: none;

  /* Even size items */
  flex-grow: 1;
  flex-shrink: 1;
  flex-basis: 0;  
}
.td11 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td11-start))),
    100% calc(100% * (1 - var(--td11-end))),
    100% 100%,
    0% 100%
  );
}


.td12 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td12-start))),
    100% calc(100% * (1 - var(--td12-end))),
    100% 100%,
    0% 100%
  );
}

.td13 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td13-start))),
    100% calc(100% * (1 - var(--td13-end))),
    100% 100%,
    0% 100%
  );
}

.td21 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td21-start))),
    100% calc(100% * (1 - var(--td21-end))),
    100% 100%,
    0% 100%
  );
}

.td22 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td22-start))),
    100% calc(100% * (1 - var(--td22-end))),
    100% 100%,
    0% 100%
  );
}

.td23 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td23-start))),
    100% calc(100% * (1 - var(--td23-end))),
    100% 100%,
    0% 100%
  );
}

.td31 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td31-start))),
    100% calc(100% * (1 - var(--td31-end))),
    100% 100%,
    0% 100%
  );
}

.td32 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td32-start))),
    100% calc(100% * (1 - var(--td32-end))),
    100% 100%,
    0% 100%
  );
}

.td33 {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;

  /* Color */
  background: var(--color, rgba(240, 50, 50, 0.75));
  clip-path: polygon(
    0% calc(100% * (1 - var(--td33-start))),
    100% calc(100% * (1 - var(--td33-end))),
    100% 100%,
    0% 100%
  );
}

.area-chart td:nth-of-type(1) {
  --color: rgba(240, 50, 50, 0.75);
}
.area-chart td:nth-of-type(2) {
  --color: rgba(255, 180, 50, 0.75);
}
.area-chart td:nth-of-type(3) {
  --color: rgba(255, 220, 90, 0.75);
}
</style>
<table class="area-chart">
  <tbody>
    <tr>
      <td class="td11"></td>
      <td class="td12"></td>
      <td  class="td13"></td>
    </tr>
    <tr>
      <td class="td21"></td>
      <td class="td22"></td>
      <td class="td23"></td>
    </tr>
    <tr>
      <td class="td31"></td>
      <td class="td32"></td>
      <td class="td33"></td>
    </tr>
  </tbody>
</table>
"""