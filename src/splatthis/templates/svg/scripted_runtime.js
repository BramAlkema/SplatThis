(function(){
  const NS = 'http://www.w3.org/2000/svg';
  const data = document.getElementById('splat-data').textContent.trim();
  const rows = data ? data.split(';') : [];
  const defs = document.getElementById('defs');
  const layer = document.getElementById('splats');
  const gradFrag = document.createDocumentFragment();
  const splatFrag = document.createDocumentFragment();
  const stops = @@STOPS@@;
  const opacityPrecision = @@OPACITY_PRECISION@@;
  const footprint = @@FOOTPRINT@@;
  const innerEnd = @@INNER_END@@;
  function addStop(grad, offset, color, opacity) {
    const stop = document.createElementNS(NS, 'stop');
    stop.setAttribute('offset', (offset * 100).toFixed(1) + '%');
    stop.setAttribute('stop-color', color);
    stop.setAttribute('stop-opacity', opacity.toFixed(opacityPrecision));
    grad.appendChild(stop);
  }
  for (let i = 0; i < rows.length; i++) {
    const v = rows[i].split(',');
    const color = 'rgb(' + v[6] + ',' + v[7] + ',' + v[8] + ')';
    const alpha = +v[9];
    const grad = document.createElementNS(NS, 'radialGradient');
    grad.id = 'g' + i;
    grad.setAttribute('cx', '50%');
    grad.setAttribute('cy', '50%');
    grad.setAttribute('r', '50%');
    grad.setAttribute('gradientUnits', 'objectBoundingBox');
    for (let j = 0; j < stops; j++) {
      const t = j / (stops - 1);
      const opacity = 1 - Math.exp(-alpha * Math.exp(-0.5 * Math.pow(t * footprint, 2)));
      addStop(grad, t * innerEnd, color, opacity);
    }
    addStop(grad, (innerEnd + 1) / 2, color, 0);
    addStop(grad, 1, color, 0);
    gradFrag.appendChild(grad);

    const ellipse = document.createElementNS(NS, 'ellipse');
    ellipse.setAttribute('cx', '0');
    ellipse.setAttribute('cy', '0');
    ellipse.setAttribute('rx', '1');
    ellipse.setAttribute('ry', '1');
    ellipse.setAttribute('transform', 'matrix(' + v[0] + ' ' + v[1] + ' ' + v[2] + ' ' + v[3] + ' ' + v[4] + ' ' + v[5] + ')');
    ellipse.setAttribute('fill', 'url(#g' + i + ')');
    ellipse.setAttribute('class', 'splat');
    splatFrag.appendChild(ellipse);
  }
  defs.appendChild(gradFrag);
  layer.appendChild(splatFrag);
  document.documentElement.setAttribute('data-rendered', String(rows.length));
})();
