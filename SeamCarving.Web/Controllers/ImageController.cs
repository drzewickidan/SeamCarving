using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using System.IO;
using SeamCarving.Web.Models;

namespace SeamCarving.Web.Controllers
{
    public class ImageController : Controller
    {
        
        [HttpGet]
        public ActionResult Upload()
        {
            return View();
        }

        [HttpPost("UploadFile")]
        public IActionResult Post(List<IFormFile> file)
        {
            List<ImageResult> images = new List<ImageResult>();
            foreach (var formFile in file)
            {
                if (formFile.Length > 0)
                {
                    using (var ms = new MemoryStream())
                    {
                        formFile.CopyTo(ms);
                        byte[] fileBytes = ms.ToArray();

                        fileBytes = SeamCarving.Program.Resize(fileBytes);
                        images.Add(new ImageResult() { FileArray = fileBytes, ContentType = formFile.ContentType });
                    }
                }
            }

            return View("Upload", images); 
        }
    }
}