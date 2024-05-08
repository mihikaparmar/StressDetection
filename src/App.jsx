import './App.css'
import Form from './components/form';
import GaugeChart from 'react-gauge-chart'
import { useState } from 'react';

function App() {
  const [val,setVal] = useState({
    score:0
  });

  return (
    <>
    <Form setVal={setVal}/>
    <GaugeChart id="gauge-chart2" percent={val.score/100}  style={{ width: 500, height: 400,margin:'auto',paddingTop:'50px'}}/>
    </>
  );
}

export default App
