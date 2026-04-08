import React, { useRef, useEffect } from 'react'

const BASE_DROPS = 120

export default function RainLayer({ active, intensity = 0 }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)

  useEffect(() => {
    if (!active) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const safeIntensity = Math.max(0, Math.min(1, Number(intensity) || 0))

    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }

    resize()
    window.addEventListener('resize', resize)

    const dropCount = Math.max(
      BASE_DROPS,
      Math.floor(BASE_DROPS * (1 + safeIntensity * 4))
    )

    const drops = Array.from({ length: dropCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      len: Math.random() * 18 + 10,
      speed: Math.random() * (2 + safeIntensity * 4) + 2,
      opacity: Math.random() * 0.5 + 0.2,
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      drops.forEach((d) => {
        ctx.beginPath()
        ctx.strokeStyle = `rgba(160, 210, 255, ${d.opacity})`
        ctx.lineWidth = 1
        ctx.moveTo(d.x, d.y)
        ctx.lineTo(d.x - 3, d.y + d.len)
        ctx.stroke()

        d.y += d.speed
        d.x -= 0.6

        if (d.y > canvas.height || d.x < -20) {
          d.y = -d.len
          d.x = Math.random() * canvas.width
        }
      })

      rafRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [active, intensity])

  if (!active) return null

  return (
    <canvas
      ref={canvasRef}
      className="rain-canvas"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5,
      }}
    />
  )
}