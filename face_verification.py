import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import tempfile
import os

FACE_API_URL = "http://localhost:7000"

def capture_and_send_image(endpoint, key_suffix):
    st.markdown("### 📸 Camera Capture")
    
    camera_input = st.camera_input(
        "Take a picture for face recognition",
        help="Make sure your face is clearly visible and well-lit",
        key=f"camera_{key_suffix}"
    )
    
    if camera_input is not None:
        try:
            st.info("📸 Processing captured image...")
            
            image = Image.open(camera_input)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save with better quality and format
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            image.save(temp_file.name, 'JPEG', quality=95, optimize=True)
            
            # Verify file was created and has content
            if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
                st.error("❌ Failed to save image properly")
                return False, "Image save failed"
            
            st.success("✅ Image captured and saved successfully!")
            st.image(image, caption="Captured Image", width=300)
            
            st.info(f"🔄 Sending to {endpoint} API...")
            st.info(f"📁 Image size: {os.path.getsize(temp_file.name)} bytes")
            
            try:
                with open(temp_file.name, 'rb') as f:
                    files = {'file': ('face.jpg', f, 'image/jpeg')}
                    
                    # Add form data if needed
                    data = {}
                    if endpoint == 'verify':
                        data['threshold'] = '0.5'
                    
                    response = requests.post(
                        f"{FACE_API_URL}/{endpoint}", 
                        files=files,
                        data=data,
                        timeout=30
                    )
                
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
                st.info(f"📡 API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        st.success(f"✅ {endpoint.title()} successful!")
                        return True, result.get('message', 'Success')
                    except:
                        st.success(f"✅ {endpoint.title()} successful!")
                        return True, "Success"
                else:
                    try:
                        error_response = response.json()
                        error_msg = error_response.get('message', f'HTTP {response.status_code} error')
                        st.error(f"❌ API Error: {error_msg}")
                        
                        # Show full response for debugging
                        with st.expander("🔍 Debug Response"):
                            st.json(error_response)
                            
                    except:
                        error_msg = f"HTTP {response.status_code} error"
                        st.error(f"❌ HTTP Error: {error_msg}")
                        
                        # Show raw response for debugging
                        with st.expander("🔍 Debug Response"):
                            st.text(response.text)
                    
                    return False, error_msg
            
            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to Face API. Make sure Flask server is running on port 7000."
                st.error(error_msg)
                return False, error_msg
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timeout. Please try again."
                st.error(error_msg)
                return False, error_msg
            except Exception as e:
                error_msg = f"❌ Request Error: {str(e)}"
                st.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"❌ Image processing error: {str(e)}"
            st.error(error_msg)
            return False, error_msg
    
    return False, "❌ No image captured. Please take a photo first."

def capture_and_verify():
    st.info("📋 Step 1: Capture your face using the camera below")
    return capture_and_send_image("verify", "verify")

def capture_and_signup():
    st.info("📋 Step 1: Capture your face to register it in the system")
    return capture_and_send_image("signUp", "signup")