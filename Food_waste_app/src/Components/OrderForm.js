import React, { useState } from 'react';
import { Box, Button, TextField, Grid, Switch, FormControlLabel, Typography, Paper, IconButton, Modal, CircularProgress } from '@mui/material';
import { CloudUpload as CloudUploadIcon, Delete as DeleteIcon } from '@mui/icons-material';
import BarChartComponent from './BarChartComponent';
 
const style = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 800,
    bgcolor: 'black',
    border: '2px solid #000',
    boxShadow: 24,
    p: 4,
};
 
const submitOrder = (data) => {
    console.log("Submitted data:", data);
    // Here you would make an API call, e.g.:
    // fetch('your-api-endpoint', { method: 'POST', body: JSON.stringify(data) })
   
};
 
const OrderForm = () => {
    const [isExcelMode, setIsExcelMode] = useState(false); // Track whether it's in Excel mode or not
    const [formData, setFormData] = useState({
        fixedLunch: '',
        additionalLunch: '',
        fixedSnacks: '',
        additionalSnacks: '',
    });
    const [files, setFiles] = useState([]); // Store uploaded files
    const [loading, setLoading] = useState(false);
    const [open, setOpen] = useState(false); //For modal
 
    // Handle the form field change
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData((prevData) => ({ ...prevData, [name]: value }));
    };
 
    // Handle toggle change
    const handleToggleChange = (event) => {
        setIsExcelMode(event.target.checked);
    };
 
    // Handle submit
    const handleSubmit = () => {
        console.log(formData)
        setLoading(true)
        //submitOrder(formData);
        setTimeout(() => {
            setLoading(false)
            setOpen(true)
        }, 4000);
    };
 
    // Handle file selection
    const handleFileChange = (e) => {
        const selectedFiles = e.target.files;
        setFiles((prevFiles) => [...prevFiles, ...Array.from(selectedFiles)]);
    };
 
    // Handle drag and drop events
    const handleDrop = (e) => {
        e.preventDefault();
        const droppedFiles = e.dataTransfer.files;
        setFiles((prevFiles) => [...prevFiles, ...Array.from(droppedFiles)]);
    };
 
    const handleDragOver = (e) => {
        e.preventDefault();
    };
 
    const handleRemoveFile = (index) => {
        setFiles(files.filter((_, i) => i !== index));
    };
 
    // Render the form
    const renderForm = () => (
        <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
            <TextField
            label="Fixed Lunch Order"
            variant="outlined"
            fullWidth
            name="fixedLunch"
            value={formData.fixedLunch}
            onChange={handleInputChange}
            slotProps={{
                input: { style: { color: 'white' } },
                inputLabel: { style: { color: 'white' } }
            }}
            sx={{
                '& .MuiOutlinedInput-root': {
                color: 'white',
                '& fieldset': { borderColor: 'white' },
                '&:hover fieldset': { borderColor: 'white' },
                '&.Mui-focused fieldset': { borderColor: 'white' }
                },
                '& label': { color: 'white' }
            }}
            />
        </Grid>
        <Grid item xs={12} sm={6}>
            <TextField
            label="Additional Lunch Ordered"
            variant="outlined"
            fullWidth
            name="additionalLunch"
            value={formData.additionalLunch}
            onChange={handleInputChange}
            slotProps={{
                input: { style: { color: 'white' } },
                inputLabel: { style: { color: 'white' } }
            }}
            sx={{
                '& .MuiOutlinedInput-root': {
                color: 'white',
                '& fieldset': { borderColor: 'white' },
                '&:hover fieldset': { borderColor: 'white' },
                '&.Mui-focused fieldset': { borderColor: 'white' }
                },
                '& label': { color: 'white' }
            }}
            />
        </Grid>
        <Grid item xs={12} sm={6}>
            <TextField
            label="Fixed Snacks Order"
            variant="outlined"
            fullWidth
            name="fixedSnacks"
            value={formData.fixedSnacks}
            onChange={handleInputChange}
            slotProps={{
                input: { style: { color: 'white' } },
                inputLabel: { style: { color: 'white' } }
            }}
            sx={{
                '& .MuiOutlinedInput-root': {
                color: 'white',
                '& fieldset': { borderColor: 'white' },
                '&:hover fieldset': { borderColor: 'white' },
                '&.Mui-focused fieldset': { borderColor: 'white' }
                },
                '& label': { color: 'white' }
            }}
            />
        </Grid>
        <Grid item xs={12} sm={6}>
            <TextField
            label="Additional Snacks Ordered"
            variant="outlined"
            fullWidth
            name="additionalSnacks"
            value={formData.additionalSnacks}
            onChange={handleInputChange}
            slotProps={{
                input: { style: { color: 'white' } },
                inputLabel: { style: { color: 'white' } }
            }}
            sx={{
                '& .MuiOutlinedInput-root': {
                color: 'white',
                '& fieldset': { borderColor: 'white' },
                '&:hover fieldset': { borderColor: 'white' },
                '&.Mui-focused fieldset': { borderColor: 'white' }
                },
                '& label': { color: 'white' }
            }}
            />
        </Grid>
        </Grid>
    );
 
    // Render the file upload UI
    const renderFileUpload = () => (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', padding: 2 }}>
        <Paper
            sx={{
            width: '100%',
            maxWidth: 400,
            padding: 3,
            borderRadius: 2,
            border: '2px dashed #1976d2',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            '&:hover': {
                backgroundColor: '#f5f5f5',
            },
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
        >
            <CloudUploadIcon sx={{ fontSize: 40, color: '#1976d2' }} />
            <Typography sx={{ fontSize: 16, color: '#1976d2', marginTop: 1 }}>
            Drag and Drop files here, or
            </Typography>
            <Button variant="contained" component="label" sx={{ marginTop: 1 }}>
            Choose Files
            <input type="file" multiple hidden onChange={handleFileChange} />
            </Button>
        </Paper>
 
        {/* Display selected files */}
        {files.length > 0 && (
            <Box sx={{ width: '100%', marginTop: 2 }}>
            <Typography variant="h6" sx={{ textAlign: 'center', marginBottom: 1 }}>
                Selected Files:
            </Typography>
            <Grid container spacing={2}>
                {files.map((file, index) => (
                <Grid item xs={12} key={index}>
                    <Paper sx={{ display: 'flex', alignItems: 'center', padding: 1.5, marginBottom: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                    <Typography variant="body1" sx={{ flex: 1 }}>
                        {file.name}
                    </Typography>
                    <IconButton onClick={() => handleRemoveFile(index)} sx={{ color: '#d32f2f' }}>
                        <DeleteIcon />
                    </IconButton>
                    </Paper>
                </Grid>
                ))}
            </Grid>
            </Box>
        )}
        </Box>
    );
 
    return (
        <Box sx={{ padding: 3 }}>
            <Modal
            open={open}
            onClose={() => setOpen(false)}
            aria-labelledby="modal-modal-title"
            aria-describedby="modal-modal-description"
            >
                <Box sx={style}>
                <Typography id="modal-modal-title" variant="h6" component="h2">
                    Actual vs Prediction
                </Typography>
                <BarChartComponent/>
                </Box>
            </Modal>
            {loading ?
                <Box sx={{ display: 'flex' }}>
                    <CircularProgress/>
                </Box>:
                <>
                <FormControlLabel control={<Switch checked={isExcelMode} onChange={handleToggleChange} />} label={isExcelMode ? "Switch to Form" : "Switch to File Upload"} />
                <br/><br/>
                {isExcelMode ? renderFileUpload() : renderForm()}
                <br/><br/>
                <Box sx={{ marginTop: 2 }}>
                    <Button variant="contained" onClick={handleSubmit}>Submit Order</Button>
                </Box>
                </>
            }
        </Box>
    );
};
 
export default OrderForm;
 