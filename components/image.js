import Image from 'next/image'

function BlogImage(image, description, width, height) {
  return <Image src={image} alt={description} width={width} height={height} />
}

export default BlogImage