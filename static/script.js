document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    for (let key in data) {
        data[key] = parseFloat(data[key]);
    }
    
    data['g-r'] = data.g - data.r;

    console.log('Sending Data :-', data)

    try{
        const response = await fetch('/predict',{
            method : 'POST',
            headers: {
                'Content-Type' : 'application/json'
            },
            body : JSON.stringify(data)
        });

        const result  = await response.json()
        console.log('Response recieveed :-', result)
        const resultBox = document.getElementById('result')

        if (result.error || result.Error) {
            resultBox.innerHTML = `<p style="color : red;"> ${result.error || result.Error}</p>`;

        } else {
            let probs = "";

            for (const [cls,prob] of Object.entries(result.probabilities)) {
                probs += `<li><b>${cls}</b>: ${(prob * 100).toFixed(2)}%</li>`;
            }

            resultBox.innerHTML = `
            <h3>Prediction : <span style="color:#58a677;">${result.prediction}</span></h3>
            <p> Class Probabilities: </p>
            <ul>${probs}</ul>
            `;

        }
        resultBox.style.display = 'block'
        
    } catch (err) {
        console.error('Failed Fetch :',err)
        alert('Error communicating with server:' + err.message);
    }
    
});