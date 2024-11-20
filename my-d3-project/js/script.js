// Example JavaScript file for D3

// Sample data
const data = [100, 200, 300, 150, 250];

// Set dimensions
const width = 500;
const height = 300;

// Create SVG container
const svg = d3.select('#chart')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

// Bind data to rectangles (bars)
svg.selectAll('rect')
    .data(data)
    .enter()
    .append('rect')
    .attr('x', (d, i) => i * 60)
    .attr('y', d => height - d)
    .attr('width', 50)
    .attr('height', d => d)
    .attr('fill', 'steelblue');


d3.json('js/data.json').then(data => {
  // Bind the loaded data to elements
  svg.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', (d, i) => i * 60)
      .attr('y', d => height - d.value)
      .attr('width', 50)
      .attr('height', d => d.value)
      .attr('fill', 'steelblue');
});
  