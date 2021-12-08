using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

[Serializable, VolumeComponentMenu("Post-processing/Custom/StyleTransferFilter")]
public sealed class StyleTransferFilter : CustomPostProcessVolumeComponent, IPostProcessComponent
{
    //public ComputeShaderParameter FilterComputeShaderParameter = new ComputeShaderParameter(null);

    [Tooltip("")]
    public Vector2IntParameter InputResolutionParameter = new Vector2IntParameter(new Vector2Int(960, 540));

    [Tooltip("")]
    public BoolParameter Inference = new BoolParameter(true);

    public bool IsActive() => true;

    public override CustomPostProcessInjectionPoint injectionPoint => CustomPostProcessInjectionPoint.AfterPostProcess;
    
    // Name of the DLL file
    const string dll = "OpenVINO_Style_Transfer_DLL";

    [DllImport(dll)]
    private static extern int FindAvailableDevices();

    [DllImport(dll)]
    private static extern IntPtr GetDeviceName(int index);

    [DllImport(dll)]
    private static extern IntPtr InitOpenVINO(string model, int width, int height, int device);

    [DllImport(dll)]
    private static extern void PerformInference(IntPtr inputData);

    [DllImport(dll)]
    private static extern void FreeResources();

    // Contains the input texture that will be sent to the OpenVINO inference engine
    private Texture2D inputTex;
    // Stores the raw pixel data for inputTex
    private byte[] inputData;

    // Unparsed list of available compute devices for OpenVINO
    private string openvinoDevices;
    // Current compute device for OpenVINO
    private string currentDevice;
    // Parsed list of compute devices for OpenVINO
    private List<string> deviceList = new List<string>();

    // File paths for the OpenVINO IR models
    private List<string> openVINOPaths = new List<string>();
    // Names of the OpenVINO IR model
    private List<string> openvinoModels = new List<string>();

    // Keeps track of whether to execute the OpenVINO model
    private bool performInference = true;

    // Used to scale the input image dimensions while maintaining aspect ratio
    private float aspectRatioScale;


    RTHandle rtHandle;


    /// <summary>
    /// Called when a model option is selected from the dropdown
    /// </summary>
    public void InitializeOpenVINO()
    {
        // Only initialize OpenVINO when performing inference
        if (performInference == false) return;

        Debug.Log("Initializing OpenVINO");
        Debug.Log($"Selected Model: {openvinoModels[0]}");
        Debug.Log($"Selected Model Path: {openVINOPaths[0]}");
        Debug.Log($"Setting Input Dims to W: {inputTex.width} x H: {inputTex.height}");
        Debug.Log("Uploading IR Model to Compute Device");

        // Set up the neural network for the OpenVINO inference engine
        currentDevice = Marshal.PtrToStringAnsi(InitOpenVINO(
            openVINOPaths[0],
            inputTex.width,
            inputTex.height,
            0));

        Debug.Log($"OpenVINO using: {currentDevice}");
    }


    /// <summary>
    /// Perform the initialization steps required when the model input is updated
    /// </summary>
    private void InitializationSteps()
    {
        // Set up the neural network for the OpenVINO inference engine
        InitializeOpenVINO();
    }

    /// <summary>
    /// Get the list of available OpenVINO models
    /// </summary>
    private void GetOpenVINOModels()
    {
        // Get the model files in each subdirectory
        List<string> openVINOFiles = new List<string>();
        openVINOFiles.AddRange(System.IO.Directory.GetFiles(Application.streamingAssetsPath + "/models"));

        // Get the paths for the .xml files for each model
        Debug.Log("Available OpenVINO Models:");
        foreach (string file in openVINOFiles)
        {
            if (file.EndsWith(".xml"))
            {
                openVINOPaths.Add(file);
                string modelName = file.Split('\\')[1];
                openvinoModels.Add(modelName.Substring(0, modelName.Length));

                Debug.Log($"Model Name: {modelName}");
                Debug.Log($"File Path: {file}");
            }
        }
        Debug.Log("");
    }



    public override void Setup()
    {
        float scale = InputResolutionParameter.value.y / (float)Screen.height;

        rtHandle = RTHandles.Alloc(
            scaleFactor: Vector2.one * scale,
            filterMode: FilterMode.Point,
            wrapMode: TextureWrapMode.Clamp,
            dimension: TextureDimension.Tex2D
            );

        inputTex = new Texture2D(rtHandle.rt.width, rtHandle.rt.height, TextureFormat.RGBA32, false);

        // Check if either the CPU of GPU is made by Intel
        string processorType = SystemInfo.processorType.ToString();
        string graphicsDeviceName = SystemInfo.graphicsDeviceName.ToString();
        if (processorType.Contains("Intel") || graphicsDeviceName.Contains("Intel"))
        {
            // Get the list of available models
            GetOpenVINOModels();

            Debug.Log("Available Devices:");
            int deviceCount = FindAvailableDevices();
            for (int i = 0; i < deviceCount; i++)
            {
                deviceList.Add(Marshal.PtrToStringAnsi(GetDeviceName(i)));
                Debug.Log(deviceList[i]);
            }
        }
        else
        {
            Inference.value = false;
            Debug.Log("No Intel hardware detected");
        }

        // Perform the requred 
        InitializationSteps();
    }

    
    /// <summary>
    /// Pin memory for the input data and send it to OpenVINO for inference
    /// </summary>
    /// <param name="inputData"></param>
    public unsafe void UpdateTexture(byte[] inputData)
    {
        //Pin Memory
        fixed (byte* p = inputData)
        {
            // Perform inference with OpenVINO
            PerformInference((IntPtr)p);
        }
    }


    public override void Render(CommandBuffer cmd, HDCamera camera, RTHandle source, RTHandle destination)
    {
        if (Inference.value)
        {
            HDUtils.BlitCameraTexture(cmd, source, rtHandle);

            RenderTexture.active = rtHandle.rt;
            inputTex.ReadPixels(new Rect(0, 0, rtHandle.rt.width, rtHandle.rt.height), 0, 0);
            inputTex.Apply();

            // Get raw data from Texture2D
            inputData = inputTex.GetRawTextureData();
            // Send reference to inputData to DLL
            UpdateTexture(inputData);

            // Load the new image data from the DLL to the texture
            inputTex.LoadRawTextureData(inputData);
            // Apply the changes to the texture
            inputTex.Apply();
            // 
            cmd.Blit(inputTex, destination, 0, 0);
        }
        else
        {
            HDUtils.BlitCameraTexture(cmd, source, destination);
        }
    }


    public override void Cleanup()
    {
        rtHandle.Release();
    }
}