const axios = require('axios');

async function postData() {
    // Your API endpoint
    const url = 'https://apps.beam.cloud/hmvz6';

    // Basic Authentication credentials
    const username = '46aa331877e16eda86991a5f585179e3';
    const password = 'e8b4b34b7400b73e82e96f22e9fed8d3';

    // Custom headers
    const headers = {
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Content-Type':'application/json'
    };

    // Data you want to send in the POST request
    const data = {
        prompt: 'a car'
    };

    try {
        const response = await axios.post(url, data, {
            headers: headers,
            auth: {
                username: username,
                password: password
            }
        });

        console.log(response.data);
    } catch (error) {
        console.error('Error posting data:', error);
    }
}

postData();