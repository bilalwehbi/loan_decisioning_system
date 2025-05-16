import React, { useState } from 'react';
import {
  Box,
  Grid,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Paper,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Description as DescriptionIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { Document } from '../../types/loan';

interface DocumentUploadFormProps {
  documents: Document[];
  onChange: (documents: Document[]) => void;
  errors?: string[];
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

const acceptedFileTypes = {
  'ID Proof': ['.pdf', '.jpg', '.jpeg', '.png'],
  'Income Proof': ['.pdf', '.jpg', '.jpeg', '.png'],
  'Bank Statement': ['.pdf'],
  'Property Documents': ['.pdf', '.jpg', '.jpeg', '.png'],
};

const DocumentUploadForm: React.FC<DocumentUploadFormProps> = ({ documents, onChange, errors = [] }) => {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const validateFile = (file: File): string | null => {
    if (file.size > MAX_FILE_SIZE) {
      return `File ${file.name} is too large. Maximum size is 10MB.`;
    }

    const extension = file.name.split('.').pop()?.toLowerCase();
    const acceptedExtensions = Object.values(acceptedFileTypes).flat();
    if (!extension || !acceptedExtensions.includes(`.${extension}`)) {
      return `File ${file.name} has an unsupported extension. Accepted types: ${acceptedExtensions.join(', ')}`;
    }

    return null;
  };

  const handleFiles = (files: FileList) => {
    const newDocuments: Document[] = [];
    const errors: string[] = [];

    Array.from(files).forEach(file => {
      const error = validateFile(file);
      if (error) {
        errors.push(error);
      } else {
        newDocuments.push({
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          type: getDocumentType(file),
          file,
        });
      }
    });

    if (errors.length > 0) {
      setError(errors.join('\n'));
    }

    if (newDocuments.length > 0) {
      onChange([...documents, ...newDocuments]);
    }
  };

  const getDocumentType = (file: File): string => {
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension === 'pdf') {
      return 'PDF Document';
    }
    if (['jpg', 'jpeg', 'png'].includes(extension || '')) {
      return 'Image';
    }
    return 'Other';
  };

  const removeDocument = (id: string) => {
    onChange(documents.filter(doc => doc.id !== id));
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Document Upload
      </Typography>
      {errors.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errors.map((error, index) => (
            <div key={index}>{error}</div>
          ))}
        </Alert>
      )}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper
            sx={{
              p: 3,
              border: '2px dashed',
              borderColor: dragActive ? 'primary.main' : 'grey.300',
              backgroundColor: dragActive ? 'action.hover' : 'background.paper',
              textAlign: 'center',
              cursor: 'pointer',
            }}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png"
              onChange={handleFileInput}
              style={{ display: 'none' }}
              id="document-upload"
            />
            <label htmlFor="document-upload">
              <Button
                component="span"
                variant="contained"
                startIcon={<CloudUploadIcon />}
                sx={{ mb: 2 }}
              >
                Upload Documents
              </Button>
            </label>
            <Typography variant="body2" color="textSecondary">
              Drag and drop files here or click to browse
            </Typography>
            <Typography variant="caption" color="textSecondary" display="block">
              Accepted file types: PDF, JPG, JPEG, PNG (Max size: 10MB)
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <List>
            {documents.map(doc => (
              <ListItem
                key={doc.id}
                secondaryAction={
                  <IconButton edge="end" onClick={() => removeDocument(doc.id)}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  <DescriptionIcon />
                </ListItemIcon>
                <ListItemText 
                  primary={doc.name} 
                  secondary={`Type: ${doc.type} (${(doc.file.size / 1024 / 1024).toFixed(2)}MB)`} 
                />
              </ListItem>
            ))}
          </List>
        </Grid>
      </Grid>
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DocumentUploadForm;
