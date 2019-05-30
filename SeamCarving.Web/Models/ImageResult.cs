using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeamCarving.Web.Models
{
    public class ImageResult
    {
        public byte[] FileArray { get; set; }
        public string ContentType { get; set; }
    }
}
