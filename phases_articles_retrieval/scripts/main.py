import click
from articles_retrieval import retrieve_and_download_articles, process_articles, keywords_gerotranscendence, keywords_solitude, gerotranscendence_dir, solitude_dir

@click.command()
@click.option('--max_results', prompt='Number of articles to be retrieved', type=int,
              help='Number of articles to be retrieved')

def retrieve_articles(max_results):
    """Retrieve and download articles based on keywords."""
    click.echo("Retrieving articles...")

    # Retrieve and download articles for both topics
    id_list_gerotranscendence = retrieve_and_download_articles(keywords_gerotranscendence, max_results=max_results)
    id_list_solitude = retrieve_and_download_articles(keywords_solitude, max_results=max_results)

    # Process articles for both topics
    articles_gerotranscendence, failed_articles_gerotranscendence = process_articles(id_list_gerotranscendence, gerotranscendence_dir)
    articles_solitude, failed_articles_solitude = process_articles(id_list_solitude, solitude_dir)

    # Display results for Gerotranscendence articles (successful first, then unsuccessful)
    click.echo("Gerotranscendence Articles:\n")
    if articles_gerotranscendence:
        click.echo("Successful Gerotranscendence Downloads:")
        for article in articles_gerotranscendence:
            click.echo(f"ID: {article[0]}, Title: {article[1]}")
            click.echo(f"PDF: {article[3]}")
    else:
        click.echo("No successful downloads for Gerotranscendence.")

    # Display unsuccessful downloads for Gerotranscendence
    if failed_articles_gerotranscendence:
        click.echo("Unsuccessful Gerotranscendence Downloads:")
        for failed in failed_articles_gerotranscendence:
            click.echo(f"ID: {failed[0]}, Title: {failed[1]}, Reason: {failed[2]}")

    # Display results for Solitude articles (successful first, then unsuccessful)
    click.echo("\nSolitude Articles:\n")
    if articles_solitude:
        click.echo("Successful Solitude Downloads:")
        for article in articles_solitude:
            click.echo(f"ID: {article[0]}, Title: {article[1]}")
            click.echo(f"PDF: {article[3]}")
    else:
        click.echo("No successful downloads for Solitude.")

    # Display unsuccessful downloads for Solitude
    if failed_articles_solitude:
        click.echo("Unsuccessful Solitude Downloads:")
        for failed in failed_articles_solitude:
            click.echo(f"ID: {failed[0]}, Title: {failed[1]}, Reason: {failed[2]}")

if __name__ == '__main__':
    retrieve_articles()
