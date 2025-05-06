from fastapi import APIRouter, Response
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os

router = APIRouter()
jinja_env = Environment(loader=FileSystemLoader('templates'))

@router.get("/properties/{id}/report/pdf")
async def export_pdf(property_id: int):
    # Get the analysis data
    data = get_analysis(property_id)
    
    # Prepare template data
    template_data = {
        'datetime': datetime,
        'kpis': data['kpis'],
        'performance_data': data['financials'],
        'executive_summary': data.get('insights', {}).get('summary', ''),
        'performance_insights': data.get('insights', {}).get('performance', []),
        'recommendations': data.get('insights', {}).get('recommendations', [])
    }
    
    # Render template
    template = jinja_env.get_template('report.html')
    html = template.render(**template_data)
    
    # Generate PDF
    pdf = HTML(string=html).write_pdf()
    
    return Response(
        content=pdf,
        media_type='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename="Property_{property_id}_Report.pdf"'
        }
    )
