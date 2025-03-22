import React, { useState } from 'react';
import './App.css'
import { Box, Tabs, Tab, Typography } from '@mui/material';
import FileUpload from './Components/FileUpload';
import OrderForm from './Components/OrderForm';
import Home from './Components/Home';
 
function a11yProps(index) {
    return {
        id: `simple-tab-${index}`,
        'aria-controls': `simple-tabpanel-${index}`,
    };
}
 
function CustomTabPanel(props) {
    const { children, value, index, ...other } = props;
 
    return (
        <div
        role="tabpanel"
        hidden={value !== index}
        id={`simple-tabpanel-${index}`}
        aria-labelledby={`simple-tab-${index}`}
        {...other}
        >
        {value === index && (
            <Box sx={{ p: 3 }}>
            {children}
            </Box>
        )}
        </div>
    );
}
 
function App() {
    const [value, setValue] = useState(0);
 
    const handleChange = (e, newValue) => {
        e.preventDefault();
        setValue(newValue);
    };
 
    return (
        <div className='tab-div'>
            <Typography variant='h4'>Sustainable Dining: Plan Better, Waste Less</Typography>
            <Typography variant='subtitle'>Predict Tomorrow’s Food Needs & Minimize Waste</Typography>
            <br/><br/>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={value} onChange={handleChange} aria-label="basic tabs example">
                <Tab label=" Home " sx={{ color: 'white', '&.Mui-selected': { color: 'lightblue' } }} {...a11yProps(0)} />
                <Tab label="Train Model" sx={{ color: 'white', '&.Mui-selected': { color: 'lightblue' } }} {...a11yProps(1)} />
                <Tab label="Prediction" sx={{ color: 'white', '&.Mui-selected': { color: 'lightblue' } }} {...a11yProps(2)} />
            </Tabs>
            <CustomTabPanel value={value} index={0}>
                <Home/>
            </CustomTabPanel>
            <CustomTabPanel value={value} index={1}>
                <FileUpload/>
            </CustomTabPanel>
            <CustomTabPanel value={value} index={2}>
                <OrderForm/>
            </CustomTabPanel>
            </Box>
        </div>
    );
}
 
export default App;