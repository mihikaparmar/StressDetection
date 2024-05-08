import React from 'react'
import { useState } from 'react';
import axios from "axios";

const form = ({setVal}) => {
    const [selectedLanguage, setSelectedLanguage] = useState('english');
    const [inputValue, setInputValue] = useState('');

    const handleLanguageChange = (event) => {
        setSelectedLanguage(event.target.value);
    };

    const handleInputChange = (event) => {
        setInputValue(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', {
                language: selectedLanguage,
                text: inputValue
            });
            setVal(response.data);
            console.log('Response:', response.data);
        } catch (error) {
            console.error('Error:', error);
        }
    };
    return (
        <div className="container">
            <h2 className="title">Language Selector</h2>
            <form onSubmit={handleSubmit} className="form">
                {/* Select field for language */}
                <div className="form-group">
                    <label htmlFor="languageSelect" className="label">Select Language:</label>
                    <select
                        id="languageSelect"
                        value={selectedLanguage}
                        onChange={handleLanguageChange}
                        className="select"
                    >
                        <option value="english">English</option>
                        <option value="hindi">Hindi</option>
                    </select>
                </div>

                {/* Input field */}
                <div className="form-group">
                    <label htmlFor="textInput" className="label">Enter Text:</label>
                    <input
                        type="text"
                        id="textInput"
                        value={inputValue}
                        onChange={handleInputChange}
                        className="input"
                    />
                </div>

                {/* Submit button */}
                <button type="submit" className="button">Submit</button>
            </form>
        </div>
    )
}

export default form