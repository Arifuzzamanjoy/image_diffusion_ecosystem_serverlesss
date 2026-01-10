"""
UI styles and CSS for the Advanced FLUX LoRA Trainer interface
"""


class UIStyles:
    """UI styles and CSS definitions"""
    
    @staticmethod
    def get_custom_css() -> str:
        """Get custom CSS for the interface"""
        return """
        .main-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .content-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header-title {
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .dataset-info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            color: white;
        }
        
        .training-section {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }
        
        .status-container {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            margin: 10px 0;
        }
        
        .upload-section {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }
        
        .train-btn {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 15px 30px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            color: white !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        
        .train-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
        }
        
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        .parameter-section {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        
        .advanced-section {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .info-box {
            background: rgba(34, 197, 94, 0.1);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 4px solid #22c55e;
        }
        
        .warning-box {
            background: rgba(251, 191, 36, 0.1);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 4px solid #fbbf24;
        }
        
        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 4px solid #ef4444;
        }
        """
    
    @staticmethod
    def get_header_html() -> str:
        """Get header HTML"""
        return """
        <div class="header-title">
            🚀 Advanced FLUX LoRA Trainer Pro
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <p style="font-size: 1.2em; color: #666;">
                Professional LoRA training with complete parameter control and captioning integration
            </p>
        </div>
        """
    
    @staticmethod
    def get_info_section_html() -> str:
        """Get information section HTML"""
        return """
        <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3>🎯 Complete FLUX LoRA Training Workflow</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
                <div>
                    <h4>1️⃣ Generate Captions</h4>
                    <ul>
                        <li>Use Advanced Image Captioning Pro</li>
                        <li>Upload ZIP of images</li>
                        <li>Download processed images ZIP</li>
                        <li>Download captions JSON</li>
                    </ul>
                </div>
                <div>
                    <h4>2️⃣ Train LoRA</h4>
                    <ul>
                        <li>Upload both files here</li>
                        <li>Configure all training parameters</li>
                        <li>Choose optimal trigger positioning</li>
                        <li>Monitor progress and results</li>
                    </ul>
                </div>
                <div>
                    <h4>3️⃣ Advanced Features</h4>
                    <ul>
                        <li>Research-optimized trigger positioning</li>
                        <li>Multiple resolution training</li>
                        <li>EMA smoothing support</li>
                        <li>Professional parameter control</li>
                    </ul>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h4>🧠 FLUX Architecture Optimizations</h4>
                <ul style="columns: 2; column-gap: 20px;">
                    <li><strong>🏆 Beginning Position:</strong> Maximum attention weight</li>
                    <li><strong>⚡ Faster Convergence:</strong> Research-proven benefits</li>
                    <li><strong>🎯 Stronger Activation:</strong> Reliable trigger response</li>
                    <li><strong>🧬 Dual Text Encoders:</strong> CLIP + T5 optimization</li>
                    <li><strong>📊 Advanced Analytics:</strong> Dataset verification</li>
                    <li><strong>🔧 Expert Controls:</strong> Professional parameters</li>
                    <li><strong>💫 Clean Spacing:</strong> Proper formatting</li>
                    <li><strong>🎨 Multiple Strategies:</strong> Flexible positioning</li>
                </ul>
            </div>
            
            <div style="margin-top: 15px; padding: 15px; background: rgba(34, 197, 94, 0.2); border-radius: 10px; border-left: 4px solid #22c55e;">
                <h4>🔬 Research Findings: Trigger Word Positioning</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 10px;">
                    <div style="background: rgba(34, 197, 94, 0.1); padding: 10px; border-radius: 8px;">
                        <strong>🏆 Beginning (Recommended)</strong>
                        <ul style="margin-top: 5px; font-size: 0.9em;">
                            <li>✅ Higher attention weights</li>
                            <li>✅ Faster training convergence</li>
                            <li>✅ Stronger style activation</li>
                            <li>✅ FLUX architecture optimized</li>
                        </ul>
                    </div>
                    <div style="background: rgba(251, 191, 36, 0.1); padding: 10px; border-radius: 8px;">
                        <strong>⚠️ End Position</strong>
                        <ul style="margin-top: 5px; font-size: 0.9em;">
                            <li>⚡ Natural language flow</li>
                            <li>❌ Lower attention weight</li>
                            <li>❌ May need more training steps</li>
                            <li>⚠️ Less reliable activation</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """