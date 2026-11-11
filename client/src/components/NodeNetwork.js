import React from 'react';
import './NodeNetwork.css';

// Node coordinates (based on a 720x860 viewBox, placed mostly on the right)
const NODES = [
  [400, 180], [560, 250], [660, 160], [470, 360], [530, 440],
  [670, 480], [450, 590], [500, 660], [640, 640], [420, 300],
];
const EDGES = [
  [0, 1], [1, 2], [0, 9], [9, 3], [1, 3], [3, 4],
  [4, 5], [4, 6], [6, 7], [5, 8], [7, 8],
];
const PULSE = [1, 4, 7];

const NodeNetwork = ({ className = '' }) => (
  <div className={`dg-network ${className}`.trim()} aria-hidden="true">
    <svg viewBox="0 0 720 860" preserveAspectRatio="xMaxYMid slice">
      {EDGES.map(([a, b]) => (
        <line key={`${a}-${b}`} x1={NODES[a][0]} y1={NODES[a][1]} x2={NODES[b][0]} y2={NODES[b][1]} />
      ))}
      {NODES.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={PULSE.includes(i) ? 5 : 3}
          className={PULSE.includes(i) ? 'dg-network__pulse' : undefined}
        />
      ))}
    </svg>
  </div>
);

export default NodeNetwork;
