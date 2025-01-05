import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import BayesNetGraph from './BayesNetGraph'
import { createTheme, MantineProvider } from '@mantine/core';
import Expectaions from './Expectaions'

function App() {
  const [count, setCount] = useState(0)

  return (
    <MantineProvider >
      <BayesNetGraph />
      <Expectaions />
    </MantineProvider>
  )
}

export default App
