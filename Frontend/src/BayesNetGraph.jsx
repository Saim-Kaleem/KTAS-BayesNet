import axios from 'axios';
import React, { useState, useEffect } from 'react';
import ReactFlow, { Controls } from 'react-flow-renderer';

const BayesNetGraph = () => {
  const [selectedNode, setSelectedNode] = useState(null);


  const [jsonData, setJsonData] = useState({ key: 'value' });  // Adjust this data based on your actual need

  
  const nodes = [
    { id: 'Sex', data: { label: 'Sex' }, position: { x: 50, y: 50 } },
    { id: 'Age', data: { label: 'Age' }, position: { x: 250, y: 50 } },
    { id: 'Arrival', data: { label: 'Arrival' }, position: { x: 50, y: 200 } },
    { id: 'Injury', data: { label: 'Injury' }, position: { x: 250, y: 200 } },
    { id: 'Pain', data: { label: 'Pain' }, position: { x: 450, y: 50 } },
    { id: 'Mental', data: { label: 'Mental' }, position: { x: 450, y: 200 } },
    { id: 'NRS_Pain', data: { label: 'NRS Pain' }, position: { x: 650, y: 125 } },
    { id: 'Chief_Complaint', data: { label: 'Chief Complaint' }, position: { x: 800, y: 50 } },
    { id: 'BP', data: { label: 'BP' }, position: { x: 50, y: 350 } },
    { id: 'HR', data: { label: 'HR' }, position: { x: 250, y: 350 } },
    { id: 'BT', data: { label: 'BT' }, position: { x: 450, y: 350 } },
    { id: 'RR', data: { label: 'RR' }, position: { x: 650, y: 350 } },
    { id: 'Saturation', data: { label: 'Saturation' }, position: { x: 850, y: 350 } },
    { id: 'KTAS_Level', data: { label: 'KTAS Level' }, position: { x: 450, y: 500 } },
  ];

  const edges = [
    { id: 'e1', source: 'Sex', target: 'Pain', animated: true },
    { id: 'e2', source: 'Arrival', target: 'Injury', animated: true },
    { id: 'e3', source: 'Age', target: 'Pain', animated: true },
    { id: 'e4', source: 'Age', target: 'Injury', animated: true },
    { id: 'e5', source: 'Age', target: 'BP', animated: true },
    { id: 'e6', source: 'Age', target: 'HR', animated: true },
    { id: 'e7', source: 'Pain', target: 'BP', animated: true },
    { id: 'e8', source: 'Pain', target: 'KTAS_Level', animated: true },
    { id: 'e9', source: 'Pain', target: 'NRS_Pain', animated: true },
    { id: 'e10', source: 'Pain', target: 'HR', animated: true },
    { id: 'e11', source: 'Pain', target: 'RR', animated: true },
    { id: 'e12', source: 'Injury', target: 'Pain', animated: true },
    { id: 'e13', source: 'Injury', target: 'Chief_Complaint', animated: true },
    { id: 'e14', source: 'Mental', target: 'Chief_Complaint', animated: true },
    { id: 'e15', source: 'Mental', target: 'KTAS_Level', animated: true },
    { id: 'e16', source: 'NRS_Pain', target: 'Chief_Complaint', animated: true },
    { id: 'e17', source: 'Chief_Complaint', target: 'KTAS_Level', animated: true },
    { id: 'e18', source: 'BP', target: 'KTAS_Level', animated: true },
    { id: 'e19', source: 'HR', target: 'Saturation', animated: true },
    { id: 'e20', source: 'RR', target: 'KTAS_Level', animated: true },
    { id: 'e21', source: 'BT', target: 'HR', animated: true },
    { id: 'e22', source: 'Saturation', target: 'KTAS_Level', animated: true },
  ];

  const onNodeClick = (event, node) => {
    setSelectedNode(node);
  };

  const closePopup = () => setSelectedNode(null);

  const renderTable = () => {
    if (!selectedNode || !jsonData) return null;

    const nodeData = jsonData[selectedNode.id];

    if (!nodeData) {
      alert(`No data available for ${selectedNode.id}`);
      return null;
    }

    const { values, variables, state_names } = nodeData;

    
  

    return (
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {variables.map((varName) => (
              <th key={varName} style={{ border: '1px solid black', padding: '8px' }}>
                {varName}
              </th>
            ))}
            <th style={{ border: '1px solid black', padding: '8px' }}>Probability</th>
          </tr>
        </thead>
        <tbody>
          {values.map((row, index) => (
            <tr key={index}>
              {row.map((value, idx) => (
                <td key={idx} style={{ border: '1px solid black', padding: '8px' }}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div style={{ height: '600px', width: '100%' }}>
      <ReactFlow nodes={nodes} edges={edges} onNodeClick={onNodeClick} fitView>
        <Controls />
      </ReactFlow>

      {selectedNode && (
        <div
          style={{
            position: 'absolute',
            top: '10%',
            left: '50%',
            transform: 'translate(-50%, -10%)',
            padding: '20px',
            background: 'white',
            border: '1px solid #ccc',
            borderRadius: '8px',
            zIndex: 1000,
            width: '80%',
          }}
        >
          <h3>Node Details</h3>
          <p><strong>ID:</strong> {selectedNode.id}</p>
          <p><strong>Label:</strong> {selectedNode.data.label}</p>
          {renderTable()}
          <button onClick={closePopup} style={{ marginTop: '10px' }}>
            Close
          </button>
        </div>
      )}
    </div>
  );
};

export default BayesNetGraph;
