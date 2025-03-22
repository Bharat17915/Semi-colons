import React, { useState } from 'react';
import { Box, Button, Typography, Paper, Grid, IconButton } from '@mui/material';
import { CloudUpload as CloudUploadIcon, Delete as DeleteIcon } from '@mui/icons-material';

function FileUpload() {
  const [files, setFiles] = useState([]);
 
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
 
  return (
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
        <Button
          variant="contained"
          component="label"
          sx={{ marginTop: 1 }}
        >
          Choose Files
          <input
            type="file"
            multiple
            hidden
            onChange={handleFileChange}
          />
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
                <Paper
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: 1.5,
                    marginBottom: 1,
                    backgroundColor: '#f5f5f5',
                    borderRadius: 1,
                  }}
                >
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
}
 
export default FileUpload;