/**
 * HTML to MathML Converter
 * 
 * Converts KaTeX-encoded math in HTML to MathML format
 * for better Word conversion
 * 
 * Usage: node katex_to_mathml.js [--input input.html] [--output output.html]
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const inputIndex = args.indexOf('--input');
const outputIndex = args.indexOf('--output');

const inputFile = inputIndex >= 0 ? args[inputIndex + 1] : 'paper_v5_final.html';
const outputFile = outputIndex >= 0 ? args[outputIndex + 1] : 'paper_with_mathml.html';

// Read input file
if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file "${inputFile}" not found`);
    console.log('Usage: node katex_to_mathml.js [--input input.html] [--output output.html]');
    process.exit(1);
}

console.log(`Reading: ${inputFile}`);
const html = fs.readFileSync(inputFile, 'utf-8');

// Extract display equations $$...$$
const displayEqRegex = /\$\$([^$]+)\$\$/g;
const displayEquations = [];
let match;

while ((match = displayEqRegex.exec(html)) !== null) {
    displayEquations.push({
        latex: match[1].trim(),
        index: displayEquations.length,
        type: 'display'
    });
}

console.log(`Found ${displayEquations.length} display equations`);

// Extract inline equations $...$ (not $$...$$)
const inlineEqRegex = /\$([^$\n]+)\$/g;
const inlineEquations = [];

while ((inlineMatch = inlineEqRegex.exec(html)) !== null) {
    const latex = inlineMatch[1].trim();
    // Skip if it's just a dollar sign or empty
    if (latex && latex !== '$') {
        inlineEquations.push({
            latex: latex,
            index: inlineEquations.length,
            type: 'inline'
        });
    }
}

console.log(`Found ${inlineEquations.length} inline equations`);

// Try to load KaTeX
let katex;
try {
    katex = require('katex');
    console.log('KaTeX loaded successfully');
} catch (e) {
    console.error('Error: KaTeX not installed. Run: npm install katex');
    console.error(e.message);
    process.exit(1);
}

// Convert display equations to MathML
const convertedDisplay = displayEquations.map(eq => {
    try {
        const mathml = katex.renderToString(eq.latex, {
            displayMode: true,
            output: 'mathml'
        });
        return { ...eq, mathml };
    } catch (e) {
        console.error(`Error converting display eq ${eq.index}: ${e.message}`);
        return { ...eq, mathml: null };
    }
});

// Convert inline equations to MathML
const convertedInline = inlineEquations.map(eq => {
    try {
        const mathml = katex.renderToString(eq.latex, {
            displayMode: false,
            output: 'mathml'
        });
        return { ...eq, mathml };
    } catch (e) {
        console.error(`Error converting inline eq ${eq.index}: ${e.message}`);
        return { ...eq, mathml: null };
    }
});

// Now create an HTML file with MathML equations
let modifiedHtml = html;

// Replace display equations first ($$...$$)
convertedDisplay.forEach(eq => {
    if (eq.mathml) {
        const mathmlHtml = `<div class="equation-mathml" data-display="true">${eq.mathml}</div>`;
        modifiedHtml = modifiedHtml.replace(
            `$$${eq.latex}$$`,
            mathmlHtml
        );
    }
});

// Replace inline equations ($...$)
modifiedHtml = modifiedHtml.replace(/\$([^$\n]+)\$/g, (fullMatch, latex) => {
    const trimmed = latex.trim();
    if (!trimmed) return fullMatch;
    
    try {
        const mathml = katex.renderToString(trimmed, {
            displayMode: false,
            output: 'mathml'
        });
        return `<span class="inline-mathml">${mathml}</span>`;
    } catch (e) {
        console.error(`Error converting inline:`, e.message);
        return fullMatch;
    }
});

// Save output
fs.writeFileSync(outputFile, modifiedHtml);
console.log(`Saved: ${outputFile}`);

// Summary
console.log('\n=== Summary ===');
console.log(`Display equations converted: ${convertedDisplay.filter(e => e.mathml).length}`);
console.log(`Inline equations converted: ${convertedInline.filter(e => e.mathml).length}`);
console.log('\nNext step: python scripts/convert_to_docx.py');
