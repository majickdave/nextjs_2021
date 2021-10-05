import { parseISO, format } from 'date-fns'

const year = Date.prototype.getFullYear();

export default function Date() {
  return <time>copyright {year} David Samuel</time>
}