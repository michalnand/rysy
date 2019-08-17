//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class ImageLoad : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal ImageLoad(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(ImageLoad obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~ImageLoad() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_ImageLoad(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public ImageLoad() : this(rysyPINVOKE.new_ImageLoad__SWIG_0(), true) {
  }

  public ImageLoad(string file_name, bool load_grayscale, bool normalise) : this(rysyPINVOKE.new_ImageLoad__SWIG_1(file_name, load_grayscale, normalise), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public ImageLoad(string file_name, bool load_grayscale) : this(rysyPINVOKE.new_ImageLoad__SWIG_2(file_name, load_grayscale), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public ImageLoad(string file_name) : this(rysyPINVOKE.new_ImageLoad__SWIG_3(file_name), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void load(string file_name, bool load_grayscale, bool normalise) {
    rysyPINVOKE.ImageLoad_load__SWIG_0(swigCPtr, file_name, load_grayscale, normalise);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void load(string file_name, bool load_grayscale) {
    rysyPINVOKE.ImageLoad_load__SWIG_1(swigCPtr, file_name, load_grayscale);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void load(string file_name) {
    rysyPINVOKE.ImageLoad_load__SWIG_2(swigCPtr, file_name);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public SWIGTYPE_p_std__vectorT_float_t get() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.ImageLoad_get(swigCPtr), false);
    return ret;
  }

  public uint width() {
    uint ret = rysyPINVOKE.ImageLoad_width(swigCPtr);
    return ret;
  }

  public uint height() {
    uint ret = rysyPINVOKE.ImageLoad_height(swigCPtr);
    return ret;
  }

  public uint channels() {
    uint ret = rysyPINVOKE.ImageLoad_channels(swigCPtr);
    return ret;
  }

  public bool grayscale() {
    bool ret = rysyPINVOKE.ImageLoad_grayscale(swigCPtr);
    return ret;
  }

  public void normalise_image(float min, float max) {
    rysyPINVOKE.ImageLoad_normalise_image__SWIG_0(swigCPtr, min, max);
  }

  public void normalise_image(float min) {
    rysyPINVOKE.ImageLoad_normalise_image__SWIG_1(swigCPtr, min);
  }

  public void normalise_image() {
    rysyPINVOKE.ImageLoad_normalise_image__SWIG_2(swigCPtr);
  }

}
